//! 配置模块
//!
//! 从环境变量加载 Router 配置，模拟 TGI 的启动参数。

use std::env;

/// Router 运行时配置
#[derive(Debug, Clone)]
pub struct RouterConfig {
    // ---- HTTP Server ----
    pub router_host: String,
    pub router_port: u16,

    // ---- Model Server ----
    pub model_server_host: String,
    pub model_server_port: u16,

    // ---- 并发控制 ----
    /// 最大并发请求数 (过载保护信号量)
    pub max_concurrent_requests: usize,

    // ---- 批处理参数 ----
    /// 单个 batch 最大请求数
    pub max_batch_size: usize,
    /// 单个 batch prefill 阶段最大 token 数
    pub max_batch_prefill_tokens: usize,
    /// 单个 batch 总 token 数上限 (prefill + decode)
    pub max_batch_total_tokens: usize,
    /// 等待队列中累计多少 token 后触发组 batch
    pub max_waiting_tokens: usize,
    /// 等待/服务比率: 控制延迟与吞吐量平衡
    /// - 较低值 (~0.1): 偏向吞吐量，等待形成更大 batch
    /// - 较高值 (~0.8): 偏向低延迟，更积极处理小 batch
    pub waiting_served_ratio: f64,

    // ---- 生成限制 ----
    pub max_input_length: usize,
    pub max_total_tokens: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            router_host: "0.0.0.0".into(),
            router_port: 3000,
            model_server_host: "127.0.0.1".into(),
            model_server_port: 50051,
            max_concurrent_requests: 128,
            max_batch_size: 32,
            max_batch_prefill_tokens: 4096,
            max_batch_total_tokens: 16384,
            max_waiting_tokens: 20,
            waiting_served_ratio: 1.2,
            max_input_length: 4096,
            max_total_tokens: 8192,
        }
    }
}

impl RouterConfig {
    /// 从环境变量加载配置
    pub fn from_env() -> anyhow::Result<Self> {
        let mut config = Self::default();

        if let Ok(v) = env::var("ROUTER_HOST") { config.router_host = v; }
        if let Ok(v) = env::var("ROUTER_PORT") { config.router_port = v.parse()?; }
        if let Ok(v) = env::var("MODEL_SERVER_HOST") { config.model_server_host = v; }
        if let Ok(v) = env::var("MODEL_SERVER_PORT") { config.model_server_port = v.parse()?; }
        if let Ok(v) = env::var("MAX_CONCURRENT_REQUESTS") { config.max_concurrent_requests = v.parse()?; }
        if let Ok(v) = env::var("MAX_BATCH_SIZE") { config.max_batch_size = v.parse()?; }
        if let Ok(v) = env::var("MAX_BATCH_PREFILL_TOKENS") { config.max_batch_prefill_tokens = v.parse()?; }
        if let Ok(v) = env::var("MAX_BATCH_TOTAL_TOKENS") { config.max_batch_total_tokens = v.parse()?; }
        if let Ok(v) = env::var("MAX_WAITING_TOKENS") { config.max_waiting_tokens = v.parse()?; }
        if let Ok(v) = env::var("WAITING_SERVED_RATIO") { config.waiting_served_ratio = v.parse()?; }
        if let Ok(v) = env::var("MAX_INPUT_LENGTH") { config.max_input_length = v.parse()?; }
        if let Ok(v) = env::var("MAX_TOTAL_TOKENS") { config.max_total_tokens = v.parse()?; }

        Ok(config)
    }
}
