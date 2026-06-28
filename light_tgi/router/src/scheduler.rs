//! Scheduler — v4 生产级 Continuous Batching 调度器
//!
//! 架构升级:
//!   v3: 每个请求 spawn 独立 Session Actor → 各自调用 StreamGenerate
//!        → "假" batch (只是并发, 不是真正的 batch forward)
//!   v4: Scheduler 维护单条 BatchStreamGenerate 双向流
//!        → 新请求通过 cmd_tx 发给 Python
//!        → Python 端做真正的 batch forward
//!        → Token 通过 token_rx 返回, Scheduler 分发到各 HTTP handler
//!
//! 优势:
//!   - 真正的 Continuous Batching: 多个请求共享同一次 GPU forward
//!   - Python 端统一调度, 最大化 GPU 利用率
//!   - 新请求随时加入, 完成的请求自动移出

use std::collections::HashMap;
use tokio::sync::mpsc;

use crate::config::RouterConfig;
use crate::event_bus::{EventBus, QueueToken, NewRequestEvent};
use crate::infer::GrpcClient;
use crate::infer::pb::{
    BatchCommand, StreamGenerateRequest, GenerationParameters,
    BatchCommandType,
};

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
    /// ★ mpsc sender: 直接将 token 发送到对应的 HTTP handler
    response_tx: mpsc::UnboundedSender<QueueToken>,
}

pub struct Scheduler {
    config: RouterConfig,
    grpc_client: GrpcClient,
    event_bus: EventBus,
    /// 等待被调度的请求
    pending: Vec<PendingRequest>,
    /// 当前活跃请求: request_id → response_tx (用于直接发送 token)
    active_requests: HashMap<String, mpsc::UnboundedSender<QueueToken>>,
}

impl Scheduler {
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
            active_requests: HashMap::new(),
        }
    }

    /// 主循环 — v4 Continuous Batching
    ///
    /// 核心变更: 不再 spawn 独立 Session Actor
    /// 而是建立单条 BatchStreamGenerate 双向流,
    /// 所有请求通过此流与 Python 端交互
    pub async fn run(mut self) {
        tracing::info!("[Scheduler v4] Continuous Batching 调度器启动");

        // 订阅新请求事件
        let mut new_request_rx = self.event_bus.subscribe_new_request();

        // ★ 双向流通道
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<BatchCommand>();
        let (token_tx, mut token_rx) = mpsc::unbounded_channel();

        // ★ spawn 双向流任务 (Rust → Python → Rust)
        let grpc = self.grpc_client.clone();
        tokio::spawn(async move {
            if let Err(e) = grpc.batch_stream_generate(cmd_rx, token_tx).await {
                tracing::error!("[Scheduler] BatchStreamGenerate 失败: {}", e);
            }
        });

        // 定时器: 周期性将 pending 请求发送给 Python
        let mut ticker = tokio::time::interval(
            tokio::time::Duration::from_millis(50)
        );

        loop {
            tokio::select! {
                // 1. 新请求到达 (HTTP)
                Ok(event) = new_request_rx.recv() => {
                    tracing::debug!("[Scheduler] 收到新请求: {}", event.request_id);
                    let NewRequestEvent {
                        request_id,
                        input_text,
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                        do_sample,
                        response_tx,
                        queue_time,
                    } = event;
                    self.pending.push(PendingRequest {
                        request_id,
                        input_text,
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                        do_sample,
                        queue_time,
                        response_tx,
                    });
                }

                // 2. 定时 flush: 将 pending 请求打包发给 Python
                _ = ticker.tick() => {
                    if !self.pending.is_empty() {
                        self.flush_pending(&cmd_tx).await;
                    }
                }

                // 3. 接收 Python 返回的 token
                Some(batch_resp) = token_rx.recv() => {
                    self.handle_batch_response(batch_resp).await;
                }
            }
        }
    }

    /// 将 pending 请求打包成 BatchCommand 发给 Python
    async fn flush_pending(
        &mut self,
        cmd_tx: &mpsc::UnboundedSender<BatchCommand>,
    ) {
        let batch_size = self.pending.len().min(self.config.max_batch_size);
        let to_send: Vec<PendingRequest> = self.pending.drain(..batch_size).collect();

        let mut new_requests = Vec::new();
        for req in &to_send {
            // ★ 保存 response_tx，用于直接将 token 发送到 HTTP handler
            self.active_requests.insert(
                req.request_id.clone(),
                req.response_tx.clone(),
            );

            new_requests.push(StreamGenerateRequest {
                request_id: req.request_id.clone(),
                input_text: req.input_text.clone(),
                params: Some(GenerationParameters {
                    max_new_tokens: req.max_new_tokens as i32,
                    temperature: req.temperature as f32,
                    top_p: req.top_p as f32,
                    top_k: req.top_k as i32,
                    repetition_penalty: 1.0,
                    seed: 0,
                    do_sample: req.do_sample,
                    stop_token_ids: vec![],
                    stop_strings: vec![],
                }),
            });
        }

        if !new_requests.is_empty() {
            let cmd = BatchCommand {
                command_type: BatchCommandType::AddRequest as i32,
                new_requests,
                cancel_request_ids: vec![],
                batch_id: 0,
            };

            if let Err(e) = cmd_tx.send(cmd) {
                tracing::error!("[Scheduler] 发送 BatchCommand 失败: {}", e);
            } else {
                tracing::info!(
                    "[Scheduler] 发送 {} 个请求到 Python (active={})",
                    to_send.len(),
                    self.active_requests.len()
                );
            }
        }
    }

    /// 处理 Python 返回的 batch token
    async fn handle_batch_response(
        &mut self,
        batch_resp: crate::infer::pb::BatchStreamResponse,
    ) {
        for token_resp in batch_resp.tokens {
            let request_id = &token_resp.request_id;

            let is_finished = token_resp.finish_reason != 0; // NONE=0
            let finish_reason = match token_resp.finish_reason {
                1 => Some("eos_token".to_string()),
                2 => Some("length".to_string()),
                3 => Some("stop_sequence".to_string()),
                4 => Some("error".to_string()),
                _ => None,
            };

            let token = token_resp.generated_token.clone().unwrap_or_default();

            let queue_token = QueueToken {
                request_id: request_id.clone(),
                token_id: token.id,
                token_text: token.text.clone(),
                is_finished,
                finish_reason: finish_reason.clone(),
            };

            // ★ 直接通过 mpsc 发送 token 到 HTTP handler (不再使用 broadcast)
            if let Some(tx) = self.active_requests.get(request_id) {
                if tx.send(queue_token).is_err() {
                    tracing::warn!("[Scheduler] 无法发送 token (receiver 已关闭): {}", request_id);
                    self.active_requests.remove(request_id);
                }
            } else {
                tracing::warn!("[Scheduler] 找不到活跃请求的 sender: {}", request_id);
            }

            // 清理已完成的请求
            if is_finished {
                self.active_requests.remove(request_id);
                tracing::debug!(
                    "[Scheduler] 请求完成: {} (reason={:?}, active={})",
                    request_id, finish_reason, self.active_requests.len()
                );
            }
        }
    }
}
