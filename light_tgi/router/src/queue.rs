//! 请求队列模块
//!
//! 使用 tokio::sync::mpsc 无界通道实现请求队列。
//! 后台 batching task 通过此队列获取待处理请求，组成 batch。
//!
//! 设计要点：
//! - 无界通道：避免因队列满而阻塞 HTTP 线程
//! - notify 机制：新请求到达时唤醒后台任务
//! - 计数器：追踪 waiting_tokens 用于组 batch 决策

use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, Notify};

/// 单个请求条目，从前台 HTTP 线程传递到后台批处理任务
#[derive(Debug)]
pub struct QueueEntry {
    /// 请求唯一 ID (UUID v4)
    pub request_id: String,
    /// 输入 token id 列表
    pub input_ids: Vec<i32>,
    /// 最大生成 token 数
    pub max_new_tokens: u32,
    /// 响应发送通道：后台任务生成 token 后通过此通道发回
    pub response_tx: mpsc::UnboundedSender<QueueToken>,
    /// 请求进入队列的时间戳 (用于统计排队延迟)
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
    /// 无界发送端 (HTTP 线程持有 clone)
    tx: mpsc::UnboundedSender<QueueEntry>,
    /// 无界接收端 (后台任务持有)
    rx: parking_lot::Mutex<Option<mpsc::UnboundedReceiver<QueueEntry>>>,
    /// 通知后台任务有新请求到达
    notify: Notify,
    /// 队列中等待的 token 总数 (原子计数器)
    waiting_tokens: AtomicUsize,
}

impl RequestQueue {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            tx,
            rx: parking_lot::Mutex::new(Some(rx)),
            notify: Notify::new(),
            waiting_tokens: AtomicUsize::new(0),
        }
    }

    /// 获取发送端的 clone (给 HTTP 线程使用)
    pub fn sender(&self) -> mpsc::UnboundedSender<QueueEntry> {
        self.tx.clone()
    }

    /// 入队操作 (HTTP 线程调用)
    /// 
    /// 1. 累加 waiting_tokens 计数
    /// 2. 将请求放入无界通道
    /// 3. 唤醒后台任务
    pub fn enqueue(&self, entry: QueueEntry) {
        let token_count = entry.input_ids.len();
        self.waiting_tokens.fetch_add(token_count, Ordering::SeqCst);
        
        // 发送到无界通道 (不会阻塞，因为是无界的)
        if let Err(e) = self.tx.send(entry) {
            tracing::error!("入队失败 (接收端已关闭): {}", e);
        }
        
        // 唤醒后台批处理任务
        self.notify.notify_one();
    }

    /// 获取当前等待 token 数
    pub fn waiting_tokens(&self) -> usize {
        self.waiting_tokens.load(Ordering::SeqCst)
    }

    /// 减少等待 token 计数 (当请求被取出组 batch 后调用)
    pub fn decrement_waiting_tokens(&self, count: usize) {
        self.waiting_tokens.fetch_sub(count, Ordering::SeqCst);
    }

    /// 等待新请求到达的通知
    pub async fn wait_for_notify(&self) {
        self.notify.notified().await;
    }

    /// 取出接收端 (后台任务启动时调用，仅一次)
    pub fn take_receiver(&self) -> Option<mpsc::UnboundedReceiver<QueueEntry>> {
        self.rx.lock().take()
    }
}
