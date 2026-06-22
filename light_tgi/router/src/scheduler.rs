//! 批处理调度器模块 (真实模型版本)
//!
//! 变更:
//!   - QueueEntry 携带原始文本 input_text
//!   - PrefillRequest 传递 input_text 给 Python Server
//!   - 采样参数从 QueueEntry 传递到 gRPC

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::config::RouterConfig;
use crate::infer::{GrpcClient, pb::*};
use crate::queue::{QueueEntry, QueueToken, RequestQueue};

pub struct BatchingScheduler {
    config: RouterConfig,
    grpc_client: GrpcClient,
    queue: Arc<RequestQueue>,
}

#[derive(Debug)]
struct RequestState {
    request_id: String,
    input_text: String,           // ★ 原始文本
    max_new_tokens: u32,
    temperature: f64,
    top_p: f64,
    top_k: u32,
    do_sample: bool,
    generated_count: u32,
    cache_handle: i64,
    response_tx: mpsc::UnboundedSender<QueueToken>,
    queue_time: chrono::DateTime<chrono::Utc>,
    is_finished: bool,
}

impl BatchingScheduler {
    pub fn new(config: RouterConfig, grpc_client: GrpcClient, queue: Arc<RequestQueue>) -> Self {
        Self { config, grpc_client, queue }
    }

    /// 后台主循环
    pub async fn run(&self) {
        tracing::info!("后台批处理任务启动 (真实模型模式)");

        let mut receiver = self.queue.take_receiver()
            .expect("队列接收端已被取出");

        let mut batch_id_counter: i64 = 0;

        loop {
            // Step 1: 等待新请求
            self.queue.wait_for_notify().await;

            // Step 2: 组 batch
            let batch_entries = self.collect_batch(&mut receiver).await;
            if batch_entries.is_empty() {
                continue;
            }

            batch_id_counter += 1;
            let batch_id = batch_id_counter;
            tracing::info!(
                "组建 Batch #{}: {} 个请求",
                batch_id, batch_entries.len()
            );

            let mut requests: Vec<RequestState> = batch_entries
                .into_iter()
                .map(|entry| RequestState {
                    request_id: entry.request_id,
                    input_text: entry.input_text,
                    max_new_tokens: entry.max_new_tokens,
                    temperature: entry.temperature,
                    top_p: entry.top_p,
                    top_k: entry.top_k,
                    do_sample: entry.do_sample,
                    generated_count: 0,
                    cache_handle: 0,
                    response_tx: entry.response_tx,
                    queue_time: entry.queue_time,
                    is_finished: false,
                })
                .collect();

            // Step 3: Prefill
            tracing::info!("Batch #{}: 开始 Prefill", batch_id);
            if let Err(e) = self.run_prefill(batch_id, &mut requests).await {
                tracing::error!("Batch #{}: Prefill 失败: {}", batch_id, e);
                self.fail_all_requests(&requests, &format!("Prefill 失败: {}", e));
                continue;
            }
            tracing::info!("Batch #{}: Prefill 完成", batch_id);

            // Step 4: Decode 循环
            let mut decode_step = 0;
            loop {
                decode_step += 1;

                let active: Vec<&RequestState> = requests.iter()
                    .filter(|r| !r.is_finished)
                    .collect();

                if active.is_empty() {
                    tracing::info!(
                        "Batch #{}: 所有请求已完成 ({} 步 decode)",
                        batch_id, decode_step - 1
                    );
                    break;
                }

                match self.run_decode(batch_id, &mut requests).await {
                    Ok(()) => {
                        for req in requests.iter_mut() {
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
                    Err(e) => {
                        tracing::error!("Batch #{}: Decode 失败: {}", batch_id, e);
                        self.fail_all_requests(&requests, &format!("Decode 失败: {}", e));
                        break;
                    }
                }
            }

            // Step 5: 清理 KV Cache
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

    /// 收集 batch
    async fn collect_batch(
        &self,
        receiver: &mut mpsc::UnboundedReceiver<QueueEntry>,
    ) -> Vec<QueueEntry> {
        let mut entries: Vec<QueueEntry> = Vec::new();

        while let Ok(entry) = receiver.try_recv() {
            if entries.len() >= self.config.max_batch_size {
                self.queue.enqueue(entry);
                break;
            }
            entries.push(entry);
        }

        // waiting_served_ratio 逻辑
        let waiting = self.queue.waiting_count();
        if !entries.is_empty() && waiting > 0 {
            let ratio = waiting as f64 / entries.len() as f64;
            if ratio < self.config.waiting_served_ratio
                && entries.len() < self.config.max_batch_size
            {
                tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                while let Ok(entry) = receiver.try_recv() {
                    if entries.len() >= self.config.max_batch_size {
                        self.queue.enqueue(entry);
                        break;
                    }
                    entries.push(entry);
                }
            }
        }

        self.queue.decrement_waiting(entries.len());
        entries
    }

    /// Prefill: 发送原始文本给 Python Server
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
                input_text: req.input_text.clone(),   // ★ 原始文本
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
                slot_ids: vec![i as i32],
                batch_id: i as i32,
            })
            .collect();

        let response = self.grpc_client.batch_prefill(prefill_requests, batch_id).await?;

        for resp in response.responses {
            if let Some(req) = requests.iter_mut().find(|r| r.request_id == resp.request_id) {
                req.cache_handle = resp.cache_handle;
                for token in resp.generated_tokens {
                    let is_eos = token.id == -1;
                    let _ = req.response_tx.send(QueueToken {
                        request_id: req.request_id.clone(),
                        token_id: token.id,
                        token_text: token.text.clone(),
                        is_finished: is_eos,
                        finish_reason: if is_eos { Some("eos_token".into()) } else { None },
                    });
                    req.generated_count += 1;
                    if is_eos {
                        req.is_finished = true;
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode 单步
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
                token_id: 0,
                cache_handle: req.cache_handle,
                params: Some(GenerationParameters {
                    max_new_tokens: (req.max_new_tokens - req.generated_count) as i32,
                    temperature: req.temperature as f32,
                    top_p: req.top_p as f32,
                    top_k: req.top_k as i32,
                    repetition_penalty: 1.0,
                    seed: 0,
                    do_sample: req.do_sample,
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
