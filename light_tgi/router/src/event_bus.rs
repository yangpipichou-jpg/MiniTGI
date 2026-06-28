//! 事件总线模块 — 架构升级核心
//!
//! 设计:
//!   - EventBus 是全局消息中枢，解耦 HTTP Server、Scheduler、Session 之间的通信
//!   - 使用 tokio::sync::broadcast 实现多播
//!   - 每个事件类型有独立的 channel，避免无关组件收到消息
//!
//! 事件流向:
//!   HTTP Request ──► [EventBus::NewRequest] ──► Scheduler Actor
//!                                                      │
//!                                              spawn Session Actor
//!                                                      │
//!                                   [EventBus::TokenGenerated] ◄─── Session
//!                                                      │
//!                                            SSE Stream ◄── HTTP

use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::{broadcast, mpsc};

// ============================================================
// 事件类型定义
// ============================================================

/// 新请求到达事件 (HTTP Server → Scheduler)
#[derive(Debug, Clone)]
pub struct NewRequestEvent {
    pub request_id: String,
    pub input_text: String,
    pub max_new_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub do_sample: bool,
    pub response_tx: mpsc::UnboundedSender<QueueToken>,
    pub queue_time: chrono::DateTime<chrono::Utc>,
}

/// Token 生成事件 (Session → HTTP Server / SSE Stream)
#[derive(Debug, Clone)]
pub struct QueueToken {
    pub request_id: String,
    pub token_id: i32,
    pub token_text: String,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
}

/// 调度指令 (Scheduler → Session)
#[derive(Debug, Clone)]
pub enum ScheduleCommand {
    /// 开始 Prefill
    StartPrefill {
        batch_id: i64,
        request_id: String,
    },
    /// 开始 Decode
    StartDecode {
        batch_id: i64,
        request_id: String,
    },
    /// 终止请求
    Abort {
        request_id: String,
        reason: String,
    },
}

/// Session 状态变更事件 (Session → Scheduler)
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Prefill 完成，携带 cache_handle
    PrefillDone {
        request_id: String,
        cache_handle: i64,
        first_token_id: i32,
        first_token_text: String,
    },
    /// Decode 一步完成
    DecodeDone {
        request_id: String,
        token_id: i32,
        token_text: String,
        is_finished: bool,
        finish_reason: Option<String>,
    },
    /// 请求出错
    Error {
        request_id: String,
        message: String,
    },
}

/// Batch 组批事件 (Scheduler 内部)
#[derive(Debug, Clone)]
pub struct BatchReadyEvent {
    pub batch_id: i64,
    pub request_ids: Vec<String>,
}

// ============================================================
// EventBus — 全局事件中枢
// ============================================================

/// 事件总线的 channel 容量
const EVENT_CHANNEL_CAPACITY: usize = 1024;

pub struct EventBus {
    /// 新请求事件: HTTP Server → Scheduler
    new_request_tx: broadcast::Sender<NewRequestEvent>,
    /// Session 事件: Session → Scheduler
    session_tx: broadcast::Sender<SessionEvent>,
    /// Token 生成事件: Session → HTTP Server (SSE)
    token_tx: broadcast::Sender<QueueToken>,
    /// 调度指令: Scheduler → Session (用 DashMap 做 point-to-point)
    /// key = request_id, value = oneshot sender
    schedule_commands: Arc<DashMap<String, mpsc::UnboundedSender<ScheduleCommand>>>,
}

impl EventBus {
    pub fn new() -> Self {
        let (new_request_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let (session_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let (token_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);

        Self {
            new_request_tx,
            session_tx,
            token_tx,
            schedule_commands: Arc::new(DashMap::new()),
        }
    }

    // ============================================================
    // NewRequest 事件 (HTTP → Scheduler)
    // ============================================================

    /// 发布新请求事件
    pub fn publish_new_request(&self, event: NewRequestEvent) {
        let _ = self.new_request_tx.send(event);
    }

    /// 订阅新请求事件 (Scheduler 使用)
    pub fn subscribe_new_request(&self) -> broadcast::Receiver<NewRequestEvent> {
        self.new_request_tx.subscribe()
    }

    // ============================================================
    // Session 事件 (Session → Scheduler)
    // ============================================================

    /// 发布 Session 事件
    pub fn publish_session_event(&self, event: SessionEvent) {
        let _ = self.session_tx.send(event);
    }

    /// 订阅 Session 事件 (Scheduler 使用)
    pub fn subscribe_session_event(&self) -> broadcast::Receiver<SessionEvent> {
        self.session_tx.subscribe()
    }

    // ============================================================
    // Token 事件 (Session → HTTP / SSE)
    // ============================================================

    /// 发布 token 事件
    pub fn publish_token(&self, token: QueueToken) {
        let _ = self.token_tx.send(token);
    }

    /// 订阅 token 事件 (HTTP Server 使用)
    pub fn subscribe_token(&self) -> broadcast::Receiver<QueueToken> {
        self.token_tx.subscribe()
    }

    // ============================================================
    // Schedule 指令 (Scheduler → Session, point-to-point)
    // ============================================================

    /// 为 request_id 注册调度命令通道
    pub fn register_schedule_channel(
        &self,
        request_id: String,
        tx: mpsc::UnboundedSender<ScheduleCommand>,
    ) {
        self.schedule_commands.insert(request_id, tx);
    }

    /// 向指定 request_id 发送调度指令
    pub fn send_schedule_command(&self, request_id: &str, cmd: ScheduleCommand) {
        if let Some(tx) = self.schedule_commands.get(request_id) {
            let _ = tx.send(cmd);
        }
    }

    /// 移除调度命令通道
    pub fn unregister_schedule_channel(&self, request_id: &str) {
        self.schedule_commands.remove(request_id);
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EventBus {
    fn clone(&self) -> Self {
        Self {
            new_request_tx: self.new_request_tx.clone(),
            session_tx: self.session_tx.clone(),
            token_tx: self.token_tx.clone(),
            schedule_commands: self.schedule_commands.clone(),
        }
    }
}
