//! 批处理调度器模块 (核心)
//!
//! 这是 Light TGI 的调度器核心，实现了 Continuous Batching 算法。
//!
//! ## 架构设计
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Foreground (API Thread)                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  POST /generate                                             │
//! │       │                                                     │
//! │       ▼                                                     │
//! │  server.rs :: generate_handler()                             │
//! │       ├─ 1. Validation (Input length, Auth)                │
//! │       ├─ 2. Semaphore::try_acquire()  ← 过载保护             │
//! │       ├─ 3. queue.enqueue(entry)       ← 入队                │
//! │       ├─ 4. notify_one()              ← 唤醒后台            │
//! │       │                                                     │
//! │       └─ return Stream<Token>        ← 立刻返回，不等待     │
//! └─────────────────────────────────────────────────────────────┘
//!                               │
//!                               │ notify
//!                               ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                 Background (Batching Task)                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  tokio::spawn(async move {                                   │
//! │      loop {                                                  │
//! │          wait_for_notify();                                 │
//! │          let batch = next_batch();       ← 组 Batch         │
//! │          if batch.is_empty() { continue; }                 │
//! │          grpc_client.prefill(batch);     ← GPU 计算         │
//! │          loop {                                              │
//! │              grpc_client.decode(batch);  ← 迭代生成         │
//! │              if batch.all_finished() { break; }            │
//! │          }                                                 │
//! │          cleanup_resources(batch);      ← 释放显存/句柄     │
//! │      }                                                     │
//! │  });                                                        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Continuous Batching (连续批处理) 详解
//!
//! 传统静态 batching 的问题：
//!   请求 A(生成100 token) 和 请求 B(生成10 token) 组成 batch
//!   即使 B 已经完成，也必须等 A 完成整个 batch 才结束
//!   → GPU 利用率低、延迟高
//!
//! Continuous Batching 的解决方案：
//!   - 请求完成时立即从 batch 中移除
//!   - 新请求可以随时加入正在运行的 batch
//!   - Prefill 和 Decode 可以在同一次 forward 中混合执行
//!
//! ## 组 Batch 策略 (next_batch)
//!
//! 基于预算 (Budget) 的批次组装：
//!
//!   prefill_token_budget = max_batch_prefill_tokens - current_decoding_tokens
//!   token_budget = max_batch_total_tokens - current_batch_tokens
//!
//! 当以下条件满足时，将新请求加入 batch：
//!   1. batch 中请求数 < max_batch_size
//!   2. 新请求的 prompt tokens <= prefill_token_budget
//!   3. 总 tokens <= token_budget
//!
//! ## 等待策略 (Waiting Served Ratio)
//!
//! waiting_served_ratio 控制延迟与吞吐量的权衡：
//!   - 比值较低 (~0.3): 等待更多请求组成大 batch → 高吞吐、高延迟
//!   - 比值较高 (~1.5): 立即处理小 batch → 低延迟、低吞吐

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::config::RouterConfig;
use crate::infer::{GrpcClient, pb::*};
use crate::queue::{QueueEntry, QueueToken, RequestQueue};

/// 批处理调度器
pub struct BatchingScheduler {
    config: RouterConfig,
    grpc_client: GrpcClient,
    queue: Arc<RequestQueue>,
}

/// 单个请求在批处理中的运行时状态
#[derive(Debug)]
struct RequestState {
    request_id: String,
    input_ids: Vec<i32>,
    max_new_tokens: u32,
    generated_count: u32,
    cache_handle: i64,
    response_tx: mpsc::UnboundedSender<QueueToken>,
    queue_time: chrono::DateTime<chrono::Utc>,
    is_finished: bool,
}

/// 一个 Batch 的运行时状态
#[derive(Debug)]
struct Batch {
    batch_id: i64,
    requests: Vec<RequestState>,
    is_prefill_done: bool,
}

impl BatchingScheduler {
    pub fn new(config: RouterConfig, grpc_client: GrpcClient, queue: Arc<RequestQueue>) -> Self {
        Self {
            config,
            grpc_client,
            queue,
        }
    }

    /// 后台主循环
    ///
    /// 永久运行的 async 循环：
    /// 1. 等待新请求通知
    /// 2. 从队列中取请求，组成 batch
    /// 3. Prefill → Decode 循环
    /// 4. 清理完成请求的资源
    pub async fn run(&self) {
        tracing::info!("后台批处理任务启动");

        // 取出队列的接收端 (仅一次)
        let mut receiver = self.queue.take_receiver()
            .expect("队列接收端已被取出");

        let mut batch_id_counter: i64 = 0;

        loop {
            // ==========================================
            // Step 1: 等待新请求到达
            // ==========================================
            tracing::debug!("等待新请求...");
            self.queue.wait_for_notify().await;

            // ==========================================
            // Step 2: 从队列收集请求，组成 batch
            // ==========================================
            let batch_entries = self.collect_batch(&mut receiver).await;

            if batch_entries.is_empty() {
                tracing::debug!("没有足够的请求组成 batch，继续等待");
                continue;
            }

            batch_id_counter += 1;
            let batch_id = batch_id_counter;
            let num_requests = batch_entries.len();
            let total_tokens: usize = batch_entries.iter().map(|e| e.input_ids.len()).sum();
            tracing::info!(
                "组建 Batch #{}: {} 个请求, {} tokens",
                batch_id, num_requests, total_tokens
            );

            // 构建 RequestState 列表
            let mut requests: Vec<RequestState> = batch_entries
                .into_iter()
                .map(|entry| RequestState {
                    request_id: entry.request_id,
                    input_ids: entry.input_ids,
                    max_new_tokens: entry.max_new_tokens,
                    generated_count: 0,
                    cache_handle: 0,
                    response_tx: entry.response_tx,
                    queue_time: entry.queue_time,
                    is_finished: false,
                })
                .collect();

            // ==========================================
            // Step 3: Prefill 阶段
            // ==========================================
            tracing::info!("Batch #{}: 开始 Prefill", batch_id);
            if let Err(e) = self.run_prefill(batch_id, &mut requests).await {
                tracing::error!("Batch #{}: Prefill 失败: {}", batch_id, e);
                self.fail_all_requests(&requests, &format!("Prefill 失败: {}", e));
                continue;
            }
            tracing::info!("Batch #{}: Prefill 完成", batch_id);

            // ==========================================
            // Step 4: Decode 循环
            // ==========================================
            let mut decode_step = 0;
            loop {
                decode_step += 1;

                // 过滤出未完成的请求
                let active: Vec<&RequestState> = requests.iter()
                    .filter(|r| !r.is_finished)
                    .collect();

                if active.is_empty() {
                    tracing::info!(
                        "Batch #{}: 所有请求已完成 (共 {} 步 decode)",
                        batch_id, decode_step - 1
                    );
                    break;
                }

                if decode_step % 10 == 1 {
                    tracing::debug!(
                        "Batch #{}: Decode step {}, {} active requests",
                        batch_id, decode_step, active.len()
                    );
                }

                // 调用 gRPC Decode
                match self.run_decode(batch_id, &mut requests).await {
                    Ok(()) => {
                        // 将生成的 token 发送给客户端
                        for req in requests.iter_mut() {
                            if req.generated_count > 0 {
                                // 这里 token 已在 run_decode 中通过 response_tx 发送
                                // 检查是否达到 max_new_tokens
                                if req.generated_count >= req.max_new_tokens && !req.is_finished {
                                    req.is_finished = true;
                                    let _ = req.response_tx.send(QueueToken {
                                        request_id: req.request_id.clone(),
                                        token_id: 0,
                                        token_text: String::new(),
                                        is_finished: true,
                                        finish_reason: Some("length".into()),
                                    });
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Batch #{}: Decode step {} 失败: {}", batch_id, decode_step, e);
                        self.fail_all_requests(&requests, &format!("Decode 失败: {}", e));
                        break;
                    }
                }
            }

            // ==========================================
            // Step 5: 清理资源 (KV cache)
            // ==========================================
            let cache_handles: Vec<i64> = requests.iter()
                .map(|r| r.cache_handle)
                .filter(|h| *h > 0)
                .collect();

            if !cache_handles.is_empty() {
                if let Err(e) = self.grpc_client.clear_cache(cache_handles).await {
                    tracing::warn!("Batch #{}: 清理缓存失败: {}", batch_id, e);
                }
            }

            tracing::info!("Batch #{}: 处理完成", batch_id);
        }
    }

    /// 从队列中收集请求组成 batch
    ///
    /// 策略：
    /// 1. 非阻塞地从 mpsc receiver 中取出所有待处理请求
    /// 2. 按预算 (budget) 筛选请求
    /// 3. 检查 waiting_served_ratio 决定是否立即组 batch
    async fn collect_batch(
        &self,
        receiver: &mut mpsc::UnboundedReceiver<QueueEntry>,
    ) -> Vec<QueueEntry> {
        let mut entries: Vec<QueueEntry> = Vec::new();
        let mut token_count: usize = 0;

        // 非阻塞地收集所有待处理请求
        while let Ok(entry) = receiver.try_recv() {
            let input_len = entry.input_ids.len();

            // 检查 token 预算
            if token_count + input_len > self.config.max_batch_prefill_tokens {
                // 预算不够，将请求放回队列
                // 简化处理：重新入队
                self.queue.enqueue(entry);
                break;
            }

            if entries.len() >= self.config.max_batch_size {
                // batch 大小已达上限，放回
                self.queue.enqueue(entry);
                break;
            }

            token_count += input_len;
            entries.push(entry);
        }

        // 检查 waiting_served_ratio
        // 如果等待请求不够多，可以继续等待
        let waiting_tokens = self.queue.waiting_tokens();
        if !entries.is_empty() && waiting_tokens > 0 {
            let ratio = waiting_tokens as f64 / token_count as f64;
            if ratio < self.config.waiting_served_ratio && entries.len() < self.config.max_batch_size {
                // 等待更多请求，但不要无限等待
                tracing::debug!(
                    "waiting_ratio={:.2} < {:.2}, 等待更多请求",
                    ratio, self.config.waiting_served_ratio
                );
                // 短暂等待
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

                // 再尝试取一次
                while let Ok(entry) = receiver.try_recv() {
                    let input_len = entry.input_ids.len();
                    if token_count + input_len > self.config.max_batch_prefill_tokens
                        || entries.len() >= self.config.max_batch_size
                    {
                        self.queue.enqueue(entry);
                        break;
                    }
                    token_count += input_len;
                    entries.push(entry);
                }
            }
        }

        // 更新等待 token 计数
        self.queue.decrement_waiting_tokens(token_count);

        entries
    }

    /// 执行 Prefill 阶段
    ///
    /// 将所有请求的 prompt 发送给 Python Server 进行预填充，
    /// 获取首 token 和 KV cache handle。
    async fn run_prefill(
        &self,
        batch_id: i64,
        requests: &mut Vec<RequestState>,
    ) -> anyhow::Result<()> {
        let prefill_requests: Vec<PrefillRequest> = requests
            .iter()
            .enumerate()
            .map(|(i, req)| PrefillRequest {
                request_id: req.request_id.clone(),
                input_ids: req.input_ids.clone(),
                params: Some(GenerationParameters {
                    max_new_tokens: req.max_new_tokens as i32,
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 0,
                    repetition_penalty: 1.0,
                    seed: 0,
                    do_sample: false,
                    stop_token_ids: vec![],
                    stop_strings: vec![],
                }),
                slot_ids: vec![i as i32],
                batch_id: i as i32,
            })
            .collect();

        let response = self.grpc_client.batch_prefill(prefill_requests, batch_id).await?;

        // 更新请求状态
        for resp in response.responses {
            if let Some(req) = requests.iter_mut().find(|r| r.request_id == resp.request_id) {
                req.cache_handle = resp.cache_handle;

                // 发送生成的 token
                for token in resp.generated_tokens {
                    let is_finished = token.id == -1; // -1 表示 EOS
                    let _ = req.response_tx.send(QueueToken {
                        request_id: req.request_id.clone(),
                        token_id: token.id,
                        token_text: token.text.clone(),
                        is_finished,
                        finish_reason: if is_finished { Some("eos_token".into()) } else { None },
                    });
                    req.generated_count += 1;

                    if is_finished {
                        req.is_finished = true;
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// 执行 Decode 阶段 (单步)
    ///
    /// 对所有活跃请求调用一次 decode，各生成一个 token。
    async fn run_decode(
        &self,
        batch_id: i64,
        requests: &mut Vec<RequestState>,
    ) -> anyhow::Result<()> {
        let decode_requests: Vec<DecodeRequest> = requests
            .iter()
            .filter(|r| !r.is_finished)
            .map(|req| DecodeRequest {
                request_id: req.request_id.clone(),
                token_id: 0, // 简化：用 0 表示"继续生成"
                cache_handle: req.cache_handle,
                params: Some(GenerationParameters {
                    max_new_tokens: (req.max_new_tokens - req.generated_count) as i32,
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 0,
                    repetition_penalty: 1.0,
                    seed: 0,
                    do_sample: false,
                    stop_token_ids: vec![],
                    stop_strings: vec![],
                }),
            })
            .collect();

        if decode_requests.is_empty() {
            return Ok(());
        }

        let response = self.grpc_client.batch_decode(decode_requests, batch_id).await?;

        for resp in response.responses {
            if let Some(req) = requests.iter_mut().find(|r| r.request_id == resp.request_id) {
                if let Some(token) = resp.generated_token {
                    let finish_reason = match resp.finish_reason() {
                        FinishReason::EosToken => Some("eos_token"),
                        FinishReason::MaxTokens => Some("length"),
                        FinishReason::StopSequence => Some("stop_sequence"),
                        _ => None,
                    };

                    let is_finished = finish_reason.is_some();

                    let _ = req.response_tx.send(QueueToken {
                        request_id: req.request_id.clone(),
                        token_id: token.id,
                        token_text: token.text.clone(),
                        is_finished,
                        finish_reason: finish_reason.map(|s| s.to_string()),
                    });

                    req.generated_count += 1;

                    if is_finished {
                        req.is_finished = true;
                    }
                }
            }
        }

        Ok(())
    }

    /// 将所有请求标记为失败
    fn fail_all_requests(&self, requests: &[RequestState], error: &str) {
        for req in requests {
            let _ = req.response_tx.send(QueueToken {
                request_id: req.request_id.clone(),
                token_id: 0,
                token_text: String::new(),
                is_finished: true,
                finish_reason: Some(format!("error: {}", error)),
            });
        }
    }
}
