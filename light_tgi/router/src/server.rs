//! HTTP Server 模块
//!
//! 职责：
//! 1. 接收客户端 HTTP POST /generate 请求
//! 2. 请求验证 (输入长度、参数合法性)
//! 3. 过载保护 (Semaphore::try_acquire)
//! 4. 入队到 RequestQueue
//! 5. 立即返回 SSE 流式响应

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
use crate::queue::{QueueEntry, QueueToken, RequestQueue};
use crate::scheduler::BatchingScheduler;

/// 应用共享状态
#[derive(Clone)]
pub struct AppState {
    pub queue: Arc<RequestQueue>,
    pub semaphore: Arc<Semaphore>,
    pub config: RouterConfig,
    pub scheduler: Arc<BatchingScheduler>,
}

// ============================================================
// HTTP 请求/响应类型
// ============================================================

/// POST /generate 请求体
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    /// 输入文本
    pub inputs: String,
    /// 生成参数 (可选)
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
}

/// SSE 流中每条消息的格式
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

/// 健康检查端点
async fn health_handler() -> &'static str {
    "OK"
}

/// 核心：POST /generate 处理函数
///
/// 流程：
/// 1. 请求验证
/// 2. 过载保护 (Semaphore)
/// 3. 创建响应通道
/// 4. 入队
/// 5. 返回 SSE 流
async fn generate_handler(
    State(state): State<AppState>,
    Json(payload): Json<GenerateRequest>,
) -> impl IntoResponse {
    let queue_time = chrono::Utc::now();

    // ---- 1. 请求验证 ----
    let validation = validate_request(&payload, &state.config);
    if let Err(err) = validation {
        return err;
    }
    let (input_ids, params) = validation.unwrap();

    // ---- 2. 过载保护 (Semaphore::try_acquire) ----
    // 如果当前并发数已达上限，立即返回 429 Too Many Requests
    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "服务器过载，请稍后重试".into(),
                    error_type: "overloaded".into(),
                }),
            )
                .into_response();
        }
    };

    // ---- 3. 创建响应通道 ----
    // 后台任务通过此通道将生成的 token 发回，我们将其转为 SSE 流
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    let request_id = uuid::Uuid::new_v4().to_string();

    // ---- 4. 入队 ----
    let entry = QueueEntry {
        request_id: request_id.clone(),
        input_ids,
        max_new_tokens: params.max_new_tokens,
        response_tx,
        queue_time,
    };
    state.queue.enqueue(entry);

    // ---- 5. 构建 SSE 流并返回 ----
    // permit 和 state 移入流中，流结束时自动释放 permit
    let stream = build_sse_stream(
        request_id,
        response_rx,
        permit,
        queue_time,
    );

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ============================================================
// 辅助函数
// ============================================================

/// 请求验证
///
/// 检查：
/// - 输入不为空
/// - 输入 token 数不超过 max_input_length
/// - 参数合法性
fn validate_request(
    req: &GenerateRequest,
    config: &RouterConfig,
) -> Result<(Vec<i32>, ValidatedParams), axum::response::Response> {
    // 检查输入非空
    if req.inputs.trim().is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "输入不能为空".into(),
                error_type: "validation_error".into(),
            }),
        )
            .into_response());
    }

    // 简化的 tokenization：按空格分词，每个词映射到模拟的 token id
    let tokens: Vec<&str> = req.inputs.split_whitespace().collect();
    let input_ids: Vec<i32> = tokens
        .iter()
        .enumerate()
        .map(|(i, _)| (i as i32 % 50000) + 1) // 模拟 token id
        .collect();

    // 检查输入长度
    if input_ids.len() > config.max_input_length {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: format!(
                    "输入过长: {} tokens (最大: {})",
                    input_ids.len(),
                    config.max_input_length
                ),
                error_type: "validation_error".into(),
            }),
        )
            .into_response());
    }

    // 提取并校验生成参数
    let max_new_tokens = req.parameters.max_new_tokens.unwrap_or(100).min(2048);
    let temperature = req.parameters.temperature.unwrap_or(1.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "temperature 必须在 0.0 ~ 2.0 之间".into(),
                error_type: "validation_error".into(),
            }),
        )
            .into_response());
    }

    // 检查总 token 数
    if input_ids.len() + max_new_tokens as usize > config.max_total_tokens {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: format!(
                    "总 token 数超限: {} (最大: {})",
                    input_ids.len() + max_new_tokens as usize,
                    config.max_total_tokens
                ),
                error_type: "validation_error".into(),
            }),
        )
            .into_response());
    }

    Ok((
        input_ids,
        ValidatedParams {
            max_new_tokens,
            temperature,
            top_p: req.parameters.top_p.unwrap_or(1.0),
            top_k: req.parameters.top_k.unwrap_or(0),
            repetition_penalty: req.parameters.repetition_penalty.unwrap_or(1.0),
            do_sample: req.parameters.do_sample.unwrap_or(true),
        },
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
///
/// 将 mpsc receiver 中的 token 转为 SSE Event 格式。
/// 流结束或出错时，permit 自动释放。
fn build_sse_stream(
    request_id: String,
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
                let queue_time_ms = (queue_time - queue_time).num_milliseconds() as u64; // 简化
                Some(StreamDetails {
                    finish_reason: token.finish_reason.unwrap_or_else(|| "length".into()),
                    generated_tokens: token_count,
                    queue_time_ms: Some(0),
                    inference_time_ms: Some(0),
                })
            } else {
                None
            },
        };

        let json = serde_json::to_string(&response).unwrap_or_default();
        Ok(Event::default().data(json))
    })
    // _permit 在此被 drop，释放信号量
}
