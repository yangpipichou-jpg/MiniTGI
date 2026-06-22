//! 请求队列模块 (真实模型版本)
//!
//! 变更:
//!   - QueueEntry 携带原始文本 input_text 而非 token ids
//!   - 新增温度/top_p 等采样参数

use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, Notify};

/// 单个请求条目，从前台 HTTP 线程传递到后台批处理任务
#[derive(Debug)]
pub struct QueueEntry {
    /// 请求唯一 ID (UUID v4)
    pub request_id: String,
    /// ★ 原始输入文本 (由 Python 端做 tokenization)
    pub input_text: String,
    /// 最大生成 token 数
    pub max_new_tokens: u32,
    /// 采样参数
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub do_sample: bool,
    /// 响应发送通道
    pub response_tx: mpsc::UnboundedSender<QueueToken>,
    /// 入队时间戳
    pub queue_time: chrono::DateTime<chrono::Utc>,
}

/// 后台任务通过此结构将 token 发回前台
#[derive(Debug, Clone)]
pub struct QueueToken {
    pub request_id: String,
    pub token_id: i32,
    pub token_text: String,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
}

/// 请求队列
pub struct RequestQueue {
    tx: mpsc::UnboundedSender<QueueEntry>,
    rx: parking_lot::Mutex<Option<mpsc::UnboundedReceiver<QueueEntry>>>,
    notify: Notify,
    /// 队列中等待的请求数 (原子计数器)
    waiting_requests: AtomicUsize,
}

impl RequestQueue {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            tx,
            rx: parking_lot::Mutex::new(Some(rx)),
            notify: Notify::new(),
            waiting_requests: AtomicUsize::new(0),
        }
    }

    /// 入队 + 唤醒后台
    pub fn enqueue(&self, entry: QueueEntry) {
        self.waiting_requests.fetch_add(1, Ordering::SeqCst);
        if let Err(e) = self.tx.send(entry) {
            tracing::error!("入队失败: {}", e);
        }
        self.notify.notify_one();
    }

    /// 当前等待的请求数
    pub fn waiting_count(&self) -> usize {
        self.waiting_requests.load(Ordering::SeqCst)
    }

    /// 减少等待计数
    pub fn decrement_waiting(&self, count: usize) {
        self.waiting_requests.fetch_sub(count, Ordering::SeqCst);
    }

    /// 等待新请求通知
    pub async fn wait_for_notify(&self) {
        self.notify.notified().await;
    }

    /// 取出接收端 (仅一次)
    pub fn take_receiver(&self) -> Option<mpsc::UnboundedReceiver<QueueEntry>> {
        self.rx.lock().take()
    }
}
