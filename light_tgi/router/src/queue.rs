//! 请求队列模块 (事件驱动版本)
//!
//! 架构升级变更:
//!   旧: 请求队列是 mpsc channel, Scheduler 直接消费
//!   新: EventBus 替代了 mpsc channel, 此模块保留数据结构定义
//!       实际的发布/订阅由 EventBus 完成

/// 单个请求条目 (保留作为数据结构)
#[derive(Debug, Clone)]
pub struct QueueEntry {
    pub request_id: String,
    pub input_text: String,
    pub max_new_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub do_sample: bool,
}

/// Token 消息 (保留作为数据结构, 实际通过 EventBus 发送)
#[derive(Debug, Clone)]
pub struct QueueToken {
    pub request_id: String,
    pub token_id: i32,
    pub token_text: String,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
}
