//! Light TGI Router - 入口文件 (事件驱动架构 v3)
//!
//! 架构升级变更:
//!   旧: server → queue(mpsc) → scheduler(同步阻塞) → grpc
//!   新: server → EventBus → Scheduler Actor → Session Actor → grpc (异步流)
//!
//! 启动流程:
//!   1. 初始化 EventBus (全局消息中枢)
//!   2. 连接 gRPC (Python Model Server)
//!   3. 启动 Scheduler Actor (tokio::spawn)
//!   4. 启动 HTTP Server (Axum)

mod server;
mod scheduler;
mod session;
mod queue;
mod infer;
mod event_bus;
mod config;

use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config::RouterConfig;
use crate::event_bus::EventBus;
use crate::infer::GrpcClient;
use crate::scheduler::SchedulerActor;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 初始化日志
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "light_tgi=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("=== Light TGI Router v3 (事件驱动架构) 启动中 ===");

    // 加载配置
    let config = RouterConfig::from_env()?;
    tracing::info!("配置: {:#?}", config);

    // 1. 连接 Python Model Server
    let grpc_addr = format!("http://{}:{}", config.model_server_host, config.model_server_port);
    tracing::info!("连接 Model Server: {}", grpc_addr);
    let grpc_client = GrpcClient::connect(&grpc_addr).await?;

    // 获取模型信息
    match grpc_client.get_model_info().await {
        Ok(info) => {
            tracing::info!(
                "模型信息: id={}, max_seq_len={}, vocab_size={}, device={}",
                info.model_id, info.max_sequence_length, info.vocab_size, info.device
            );
        }
        Err(e) => {
            tracing::warn!("获取模型信息失败: {}，继续启动...", e);
        }
    }

    // 2. ★ 创建 EventBus (全局消息中枢)
    let event_bus = EventBus::new();

    // 3. ★ 创建 Scheduler Actor
    let scheduler = SchedulerActor::new(
        config.clone(),
        grpc_client.clone(),
        event_bus.clone(),
    );

    // 4. ★ spawn Scheduler Actor (异步、独立运行)
    tokio::spawn(async move {
        scheduler.run().await;
    });
    tracing::info!("Scheduler Actor 已启动");

    // 5. 创建并发控制信号量 (过载保护)
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));

    // 6. 构建 HTTP 路由并启动
    let app_state = server::AppState {
        event_bus: event_bus.clone(),
        semaphore: semaphore.clone(),
        config: config.clone(),
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
