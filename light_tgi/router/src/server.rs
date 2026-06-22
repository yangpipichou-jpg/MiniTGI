//! HTTP Server 模块 (真实模型版本)
//!
//! 变更:
//!   - Router 不再做 tokenization，传原始文本给 Python Server
//!   - 使用 gRPC Tokenize RPC 做输入长度验证
//!   - 保留过载保护和 SSE 流式输出

use std::sync::Arc;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::post,
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::config::RouterConfig;
use crate::infer::GrpcClient;
use crate::queue::{QueueEntry, QueueToken, RequestQueue};
use crate::scheduler::BatchingScheduler;

/// 应用共享状态
#[derive(Clone)]
pub struct AppState {
    pub queue: Arc<RequestQueue>,
    pub semaphore: Arc<Semaphore>,
    pub config: RouterConfig,
    pub scheduler: Arc<BatchingScheduler>,
    pub grpc_client: GrpcClient,
}

// ============================================================
// HTTP 请求/响应类型
// ============================================================

/// POST /generate 请求体 (兼容 TGI API 格式)
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub inputs: String,
    #[serde(default)]
    pub parameters: GenerateParameters,
}

#[derive(Debug, Deserialize, Default)]
pub struct GenerateParameters {
    pub max_new_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f64>,
    pub do_sample: Option<bool>,
    pub stop_sequences: Option<Vec<String>>,
    pub seed: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct GenerateStreamResponse {
    pub token: TokenInfo,
    pub generated_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<StreamDetails>,
}

#[derive(Debug, Serialize)]
pub struct TokenInfo {
    pub id: i32,
    pub text: String,
    pub special: bool,
}

#[derive(Debug, Serialize)]
pub struct StreamDetails {
    pub finish_reason: String,
    pub generated_tokens: u32,
    pub queue_time_ms: Option<u64>,
    pub inference_time_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: String,
}

// ============================================================
// 构建路由
// ============================================================

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/generate", post(generate_handler))
        .route("/health", axum::routing::get(health_handler))
        .with_state(state)
}

// ============================================================
// Handler 实现
// ============================================================

async fn health_handler() -> &'static str {
    "OK"
}

/// 核心: POST /generate
async fn generate_handler(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> impl IntoResponse {
    let queue_time = chrono::Utc::now();

    // ---- 1. 请求验证 (使用 gRPC Tokenize 做真实 token 计数) ----
    let validation = validate_request(&payload, &state).await;
    if let Err(err) = validation {
        return err;
    }
    let (params, input_text) = validation.unwrap();

    // ---- 2. 过载保护 ----
    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "服务器过载，请稍后重试".into(),
                    error_type: "overloaded".into(),
                }),
            ).into_response();
        }
    };

    // ---- 3. 创建响应通道 ----
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let request_id = uuid::Uuid::new_v4().to_string();

    // ---- 4. 入队 (传递原始文本) ----
    let entry = QueueEntry {
        request_id: request_id.clone(),
        input_text: input_text,    // ★ 原始文本
        max_new_tokens: params.max_new_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        do_sample: params.do_sample,
        response_tx,
        queue_time,
    };
    state.queue.enqueue(entry);

    // ---- 5. 返回 SSE 流 ----
    let stream = build_sse_stream(request_id, response_rx, permit, queue_time);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ============================================================
// 请求验证 (使用 gRPC Tokenize)
// ============================================================

async fn validate_request(
    req: &GenerateRequest,
    state: &AppState,
) -> Result<(ValidatedParams, String), axum::response::Response> {
    // 检查输入非空
    let input_text = req.inputs.trim().to_string();
    if input_text.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "输入不能为空".into(),
                error_type: "validation_error".into(),
            }),
        ).into_response());
    }

    // ★ 使用 gRPC Tokenize 获取真实 token 数
    let token_count = match state.grpc_client.tokenize(&input_text).await {
        Ok(count) => count,
        Err(e) => {
            tracing::warn!("Tokenize 失败: {}，跳过长度验证", e);
            // 如果 tokenize 失败，允许请求通过 (稍后在 Python 端会再次验证)
            input_text.len() / 4  // 粗略估计: 英文约 4 字符/token
        }
    };

    // 检查输入长度
    if token_count > state.config.max_input_length {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: format!(
                    "输入过长: {} tokens (最大: {})",
                    token_count, state.config.max_input_length
                ),
                error_type: "validation_error".into(),
            }),
        ).into_response());
    }

    // 参数校验
    let max_new_tokens = req.parameters.max_new_tokens.unwrap_or(100).min(2048);
    let temperature = req.parameters.temperature.unwrap_or(1.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "temperature 必须在 0.0 ~ 2.0 之间".into(),
                error_type: "validation_error".into(),
            }),
        ).into_response());
    }

    if token_count + max_new_tokens as usize > state.config.max_total_tokens {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: format!(
                    "总 token 数超限: {} (最大: {})",
                    token_count + max_new_tokens as usize,
                    state.config.max_total_tokens
                ),
                error_type: "validation_error".into(),
            }),
        ).into_response());
    }

    Ok((
        ValidatedParams {
            max_new_tokens,
            temperature,
            top_p: req.parameters.top_p.unwrap_or(1.0),
            top_k: req.parameters.top_k.unwrap_or(0),
            repetition_penalty: req.parameters.repetition_penalty.unwrap_or(1.0),
            do_sample: req.parameters.do_sample.unwrap_or(false),
        },
        input_text,
    ))
}

struct ValidatedParams {
    max_new_tokens: u32,
    temperature: f64,
    top_p: f64,
    top_k: u32,
    repetition_penalty: f64,
    do_sample: bool,
}

/// 构建 SSE 流
fn build_sse_stream(
    _request_id: String,
    response_rx: mpsc::UnboundedReceiver<QueueToken>,
    _permit: tokio::sync::OwnedSemaphorePermit,
    queue_time: chrono::DateTime<chrono::Utc>,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let stream = UnboundedReceiverStream::new(response_rx);
    let mut generated_text = String::new();
    let mut token_count: u32 = 0;

    stream.map(move |token| {
        token_count += 1;
        generated_text.push_str(&token.token_text);

        let response = GenerateStreamResponse {
            token: TokenInfo {
                id: token.token_id,
                text: token.token_text.clone(),
                special: false,
            },
            generated_text: Some(generated_text.clone()),
            details: if token.is_finished {
                let now = chrono::Utc::now();
                let queue_ms = (now - queue_time).num_milliseconds() as u64;
                Some(StreamDetails {
                    finish_reason: token.finish_reason.unwrap_or_else(|| "length".into()),
                    generated_tokens: token_count,
                    queue_time_ms: Some(queue_ms),
                    inference_time_ms: Some(0),
                })
            } else {
                None
            },
        };

        let json = serde_json::to_string(&response).unwrap_or_default();
        Ok(Event::default().data(json))
    })
}
