//! Light TGI Router - 入口文件
//!
//! 职责：
//! 1. 初始化配置和日志
//! 2. 创建 gRPC 客户端连接到 Python Model Server
//! 3. 启动 HTTP Server 和后台 Batching Task
//! 4. 优雅关闭

mod server;
mod scheduler;
mod queue;
mod infer;
mod config;

use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config::RouterConfig;
use crate::infer::GrpcClient;
use crate::queue::RequestQueue;
use crate::scheduler::BatchingScheduler;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化日志
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "light_tgi=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("=== Light TGI Router 启动中 ===");

    // 加载配置
    let config = RouterConfig::from_env()?;
    tracing::info!("配置: {:#?}", config);

    // 1. 创建 gRPC 客户端连接 Python Model Server
    let grpc_addr = format!("http://{}:{}", config.model_server_host, config.model_server_port);
    tracing::info!("连接 Model Server: {}", grpc_addr);
    let grpc_client = GrpcClient::connect(&grpc_addr).await?;

    // 获取模型信息
    let model_info = grpc_client.get_model_info().await?;
    tracing::info!(
        "模型信息: id={}, max_seq_len={}, vocab_size={}",
        model_info.model_id,
        model_info.max_sequence_length,
        model_info.vocab_size
    );

    // 2. 创建请求队列 (无界通道)
    let queue = Arc::new(RequestQueue::new());

    // 3. 创建并发控制信号量 (过载保护)
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

    // 4. 创建调度器
    let scheduler = Arc::new(BatchingScheduler::new(
        config.clone(),
        grpc_client.clone(),
        queue.clone(),
    ));

    // 5. 启动后台批处理任务 (spawn 一个永不结束的 tokio task)
    let scheduler_clone = scheduler.clone();
    tokio::spawn(async move {
        scheduler_clone.run().await;
    });

    // 6. 构建 HTTP 路由并启动
    let app_state = server::AppState {
        queue: queue.clone(),
        semaphore: semaphore.clone(),
        config: config.clone(),
        scheduler: scheduler.clone(),
        grpc_client: grpc_client.clone(),
    };

    let app = server::build_router(app_state);

    let bind_addr = format!("{}:{}", config.router_host, config.router_port);
    tracing::info!("HTTP Server 监听: {}", bind_addr);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Router 已关闭");
    Ok(())
}

/// 优雅关闭信号处理
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("无法监听 Ctrl+C");
    tracing::info!("收到关闭信号，正在优雅退出...");
}
