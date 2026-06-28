//! Scheduler Actor — 事件驱动的批处理调度器
//!
//! 架构升级核心变更:
//!   旧: scheduler.run() 直接同步调用 grpc，阻塞等待 → 整个 batch 串行
//!   新: scheduler 是 Actor，订阅 EventBus 事件:
//!       1. 收到 NewRequestEvent → 积累到 pending 队列
//!       2. 按预算组 batch → 为每个请求 spawn Session Actor
//!       3. 向 Session 发送 StartPrefill / StartDecode 指令
//!       4. 收到 SessionEvent → 决定下一步调度
//!
//! 优势:
//!   - 调度器不再阻塞在 gRPC 调用上
//!   - 多个 Session 可以并发执行 (真正的 Continuous Batching)
//!   - Scheduler 只做协调，不做执行

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::config::RouterConfig;
use crate::event_bus::{EventBus, NewRequestEvent, ScheduleCommand, SessionEvent};
use crate::infer::GrpcClient;
use crate::session::Session;

/// 调度器内部维护的请求元信息
#[derive(Debug)]
struct PendingRequest {
    request_id: String,
    input_text: String,
    max_new_tokens: u32,
    temperature: f64,
    top_p: f64,
    top_k: u32,
    do_sample: bool,
    queue_time: chrono::DateTime<chrono::Utc>,
}

/// Batch 状态
#[derive(Debug)]
struct ActiveBatch {
    batch_id: i64,
    /// batch 中的 request_id → 当前步数
    requests: HashMap<String, u32>,
}

pub struct SchedulerActor {
    config: RouterConfig,
    grpc_client: GrpcClient,
    event_bus: EventBus,
    /// 等待被调度的请求
    pending: Vec<PendingRequest>,
    /// 当前活跃的 batch
    active_batches: HashMap<i64, ActiveBatch>,
    batch_id_counter: i64,
}

impl SchedulerActor {
    pub fn new(
        config: RouterConfig,
        grpc_client: GrpcClient,
        event_bus: EventBus,
    ) -> Self {
        Self {
            config,
            grpc_client,
            event_bus,
            pending: Vec::new(),
            active_batches: HashMap::new(),
            batch_id_counter: 0,
        }
    }

    /// Actor 主循环 — 事件驱动
    pub async fn run(mut self) {
        tracing::info!("[Scheduler] 事件驱动调度器启动");

        // 订阅事件
        let mut new_request_rx = self.event_bus.subscribe_new_request();
        let mut session_event_rx = self.event_bus.subscribe_session_event();

        // 定时器: 周期性检查 pending 队列，即使没有新请求也触发组 batch
        let mut ticker = tokio::time::interval(
            tokio::time::Duration::from_millis(50) // 每 50ms 检查一次
        );

        loop {
            tokio::select! {
                // 1. 新请求到达
                Ok(event) = new_request_rx.recv() => {
                    tracing::debug!(
                        "[Scheduler] 收到新请求: {}",
                        event.request_id
                    );
                    self.pending.push(PendingRequest {
                        request_id: event.request_id,
                        input_text: event.input_text,
                        max_new_tokens: event.max_new_tokens,
                        temperature: event.temperature,
                        top_p: event.top_p,
                        top_k: event.top_k,
                        do_sample: event.do_sample,
                        queue_time: event.queue_time,
                    });
                }

                // 2. Session 状态变更
                Ok(event) = session_event_rx.recv() => {
                    self.handle_session_event(event).await;
                }

                // 3. 定时检查 pending 队列
                _ = ticker.tick() => {
                    if !self.pending.is_empty() {
                        self.try_schedule_batch().await;
                    }
                }
            }
        }
    }

    /// 处理 Session 发来的事件
    async fn handle_session_event(&mut self, event: SessionEvent) {
        match &event {
            SessionEvent::PrefillDone { request_id, .. } => {
                tracing::debug!("[Scheduler] PrefillDone: {}", request_id);
                // Prefill 完成 → 立即发 StartDecode
                self.event_bus.send_schedule_command(
                    request_id,
                    ScheduleCommand::StartDecode {
                        batch_id: 0,  // 简化: batch_id 透传
                        request_id: request_id.clone(),
                    },
                );
            }

            SessionEvent::DecodeDone { request_id, is_finished, .. } => {
                if *is_finished {
                    tracing::debug!("[Scheduler] 请求完成: {}", request_id);
                    // 清理 batch 中的记录
                    for batch in self.active_batches.values_mut() {
                        batch.requests.remove(request_id);
                    }
                    // 移除空的 batch
                    self.active_batches.retain(|_, b| !b.requests.is_empty());
                } else {
                    // 继续发 StartDecode
                    self.event_bus.send_schedule_command(
                        request_id,
                        ScheduleCommand::StartDecode {
                            batch_id: 0,
                            request_id: request_id.clone(),
                        },
                    );
                }
            }

            SessionEvent::Error { request_id, message } => {
                tracing::error!("[Scheduler] Session 错误: {} - {}", request_id, message);
                for batch in self.active_batches.values_mut() {
                    batch.requests.remove(request_id);
                }
                self.active_batches.retain(|_, b| !b.requests.is_empty());
            }
        }
    }

    /// 尝试将 pending 请求组装成 batch 并启动 Session
    async fn try_schedule_batch(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        // 按预算收集请求 (与旧版相同的预算逻辑)
        let batch_size = self.pending.len().min(self.config.max_batch_size);
        let to_schedule: Vec<PendingRequest> = self.pending.drain(..batch_size).collect();

        self.batch_id_counter += 1;
        let batch_id = self.batch_id_counter;

        tracing::info!(
            "[Scheduler] 组建 Batch #{}: {} 个请求",
            batch_id, to_schedule.len()
        );

        let mut request_ids = Vec::new();

        for req in to_schedule {
            let request_id = req.request_id.clone();

            // 为每个请求创建独立的调度命令通道
            let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
            self.event_bus.register_schedule_channel(request_id.clone(), cmd_tx);

            // 创建 Session Actor
            let session = Session::new(
                request_id.clone(),
                req.input_text,
                req.max_new_tokens,
                req.temperature,
                req.top_p,
                req.top_k,
                req.do_sample,
                self.grpc_client.clone(),
                self.event_bus.clone(),
                cmd_rx,
                req.queue_time,
            );

            // ★ spawn Session Actor (异步、独立生命周期)
            tokio::spawn(async move {
                session.run().await;
            });

            request_ids.push(request_id);
        }

        // 记录活跃 batch
        self.active_batches.insert(
            batch_id,
            ActiveBatch {
                batch_id,
                requests: request_ids.iter().map(|id| (id.clone(), 0)).collect(),
            },
        );

        // ★ 向所有 Session 发送 StartPrefill 指令 (非阻塞!)
        for request_id in &request_ids {
            self.event_bus.send_schedule_command(
                request_id,
                ScheduleCommand::StartPrefill {
                    batch_id,
                    request_id: request_id.clone(),
                },
            );
        }
    }
}
