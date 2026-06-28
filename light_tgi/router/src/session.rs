//! Session Actor — 每个请求独立的生命周期管理
//!
//! 架构升级核心变更:
//!   旧: scheduler 直接同步调用 grpc_client.prefill() / grpc_client.decode()，阻塞整个 batch
//!   新: 每个请求 spawn 独立 Session Actor，通过 EventBus 异步通信
//!
//! Session 状态机:
//!   CREATED → PREFILLING → DECODING → FINISHED
//!                  ↓            ↓
//!               ERROR        ERROR

use tokio::sync::mpsc;

use crate::event_bus::{EventBus, QueueToken, ScheduleCommand, SessionEvent};
use crate::infer::GrpcClient;

/// Session 内部状态
#[derive(Debug, Clone, PartialEq)]
enum SessionState {
    Created,
    Prefilling,
    Decoding,
    Finished,
    Errored,
}

/// Session Actor — 管理单个推理请求的完整生命周期
pub struct Session {
    request_id: String,
    input_text: String,
    max_new_tokens: u32,
    temperature: f64,
    top_p: f64,
    top_k: u32,
    do_sample: bool,
    state: SessionState,
    cache_handle: i64,
    generated_count: u32,
    grpc_client: GrpcClient,
    event_bus: EventBus,
    /// 接收 Scheduler 发来的调度指令
    cmd_rx: mpsc::UnboundedReceiver<ScheduleCommand>,
    queue_time: chrono::DateTime<chrono::Utc>,
}

impl Session {
    /// 创建新 Session (在 CREATED 状态)
    pub fn new(
        request_id: String,
        input_text: String,
        max_new_tokens: u32,
        temperature: f64,
        top_p: f64,
        top_k: u32,
        do_sample: bool,
        grpc_client: GrpcClient,
        event_bus: EventBus,
        cmd_rx: mpsc::UnboundedReceiver<ScheduleCommand>,
        queue_time: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            request_id,
            input_text,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            state: SessionState::Created,
            cache_handle: 0,
            generated_count: 0,
            grpc_client,
            event_bus,
            cmd_rx,
            queue_time,
        }
    }

    /// Session 主循环 — 等待 Scheduler 指令并执行
    pub async fn run(mut self) {
        tracing::debug!("[Session] {} 启动, 等待调度指令...", self.request_id);

        while self.state != SessionState::Finished && self.state != SessionState::Errored {
            match self.cmd_rx.recv().await {
                Some(ScheduleCommand::StartPrefill { batch_id, .. }) => {
                    self.state = SessionState::Prefilling;
                    if let Err(e) = self.do_prefill().await {
                        tracing::error!(
                            "[Session] {} Prefill 失败: {}",
                            self.request_id, e
                        );
                        self.state = SessionState::Errored;
                        self.event_bus.publish_session_event(SessionEvent::Error {
                            request_id: self.request_id.clone(),
                            message: format!("Prefill 失败: {}", e),
                        });
                        break;
                    }
                    tracing::info!(
                        "[Session] {} Prefill 完成 (batch={})",
                        self.request_id, batch_id
                    );
                }

                Some(ScheduleCommand::StartDecode { .. }) => {
                    self.state = SessionState::Decoding;
                    if let Err(e) = self.do_decode().await {
                        tracing::error!(
                            "[Session] {} Decode 失败: {}",
                            self.request_id, e
                        );
                        self.state = SessionState::Errored;
                        self.event_bus.publish_session_event(SessionEvent::Error {
                            request_id: self.request_id.clone(),
                            message: format!("Decode 失败: {}", e),
                        });
                        break;
                    }
                }

                Some(ScheduleCommand::Abort { reason, .. }) => {
                    tracing::warn!(
                        "[Session] {} 被终止: {}",
                        self.request_id, reason
                    );
                    self.state = SessionState::Errored;
                    // 通知 HTTP 端
                    self.event_bus.publish_token(QueueToken {
                        request_id: self.request_id.clone(),
                        token_id: 0,
                        token_text: String::new(),
                        is_finished: true,
                        finish_reason: Some(format!("aborted: {}", reason)),
                    });
                    break;
                }

                None => {
                    // channel 关闭，退出
                    tracing::debug!("[Session] {} 指令通道关闭，退出", self.request_id);
                    break;
                }
            }
        }

        // 清理: 从 EventBus 注销
        self.event_bus.unregister_schedule_channel(&self.request_id);

        // 如果正常结束但还没发 finish token
        if self.state == SessionState::Decoding {
            self.event_bus.publish_token(QueueToken {
                request_id: self.request_id.clone(),
                token_id: 0,
                token_text: String::new(),
                is_finished: true,
                finish_reason: Some("length".into()),
            });
        }

        tracing::debug!(
            "[Session] {} 退出, 状态={:?}, 生成={} tokens",
            self.request_id, self.state, self.generated_count
        );
    }

    /// 执行 Prefill: 发送原始文本给 Python Server
    async fn do_prefill(&mut self) -> anyhow::Result<()> {
        use crate::infer::pb::*;

        let prefill_req = PrefillRequest {
            request_id: self.request_id.clone(),
            input_text: self.input_text.clone(),
            params: Some(self.build_params()),
            slot_ids: vec![],
            batch_id: 0,
        };

        let response = self.grpc_client
            .batch_prefill(vec![prefill_req], 0)
            .await?;

        if let Some(resp) = response.responses.into_iter().next() {
            self.cache_handle = resp.cache_handle;

            // 发送首 token 到 HTTP 端 (通过 EventBus)
            for token in resp.generated_tokens {
                let is_eos = token.id == -1;
                self.generated_count += 1;

                self.event_bus.publish_token(QueueToken {
                    request_id: self.request_id.clone(),
                    token_id: token.id,
                    token_text: token.text.clone(),
                    is_finished: is_eos,
                    finish_reason: if is_eos { Some("eos_token".into()) } else { None },
                });

                if is_eos {
                    self.state = SessionState::Finished;
                    return Ok(());
                }
            }

            // 通知 Scheduler: Prefill 完成
            self.event_bus.publish_session_event(SessionEvent::PrefillDone {
                request_id: self.request_id.clone(),
                cache_handle: self.cache_handle,
                first_token_id: 0, // 简化处理
                first_token_text: String::new(),
            });
        }

        Ok(())
    }

    /// 执行一次 Decode: 生成下一个 token
    async fn do_decode(&mut self) -> anyhow::Result<()> {
        use crate::infer::pb::*;

        if self.generated_count >= self.max_new_tokens {
            self.state = SessionState::Finished;
            self.event_bus.publish_session_event(SessionEvent::DecodeDone {
                request_id: self.request_id.clone(),
                token_id: 0,
                token_text: String::new(),
                is_finished: true,
                finish_reason: Some("length".into()),
            });
            return Ok(());
        }

        let decode_req = DecodeRequest {
            request_id: self.request_id.clone(),
            token_id: 0,
            cache_handle: self.cache_handle,
            params: Some(self.build_params()),
        };

        let response = self.grpc_client
            .batch_decode(vec![decode_req], 0)
            .await?;

        for resp in response.responses {
            if let Some(ref token) = resp.generated_token {
                self.generated_count += 1;

                let finish_reason = match resp.finish_reason() {
                    FinishReason::EosToken => Some("eos_token"),
                    FinishReason::MaxTokens => Some("length"),
                    FinishReason::StopSequence => Some("stop_sequence"),
                    _ => None,
                };
                let is_finished = finish_reason.is_some();

                // 发送 token 到 HTTP 端 (通过 EventBus)
                self.event_bus.publish_token(QueueToken {
                    request_id: self.request_id.clone(),
                    token_id: token.id,
                    token_text: token.text.clone(),
                    is_finished,
                    finish_reason: finish_reason.map(|s| s.to_string()),
                });

                // 通知 Scheduler
                self.event_bus.publish_session_event(SessionEvent::DecodeDone {
                    request_id: self.request_id.clone(),
                    token_id: token.id,
                    token_text: token.text.clone(),
                    is_finished,
                    finish_reason: finish_reason.map(|s| s.to_string()),
                });

                if is_finished {
                    self.state = SessionState::Finished;
                }
            }
        }

        Ok(())
    }

    fn build_params(&self) -> crate::infer::pb::GenerationParameters {
        crate::infer::pb::GenerationParameters {
            max_new_tokens: (self.max_new_tokens - self.generated_count) as i32,
            temperature: self.temperature as f32,
            top_p: self.top_p as f32,
            top_k: self.top_k as i32,
            repetition_penalty: 1.0,
            seed: 0,
            do_sample: self.do_sample,
            stop_token_ids: vec![],
            stop_strings: vec![],
        }
    }
}
