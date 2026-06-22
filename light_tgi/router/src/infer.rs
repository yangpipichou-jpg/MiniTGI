//! gRPC 推理客户端模块
//!
//! 封装与 Python Model Server 的 gRPC 通信。
//! 提供 Prefill、Decode、ClearCache 等 RPC 调用。
//!
//! 设计要点：
//! - tonic 自动生成的 client 代码
//! - 连接重试机制
//! - 超时控制

use std::sync::Arc;
use tonic::transport::Channel;

// 由 build.rs 从 proto 文件生成
pub mod pb {
    tonic::include_proto!("light_tgi");
}

use pb::{
    text_generation_service_client::TextGenerationServiceClient,
    BatchPrefillRequest, BatchPrefillResponse,
    BatchDecodeRequest, BatchDecodeResponse,
    ClearCacheRequest, ClearCacheResponse,
    ModelInfoRequest, ModelInfoResponse,
    PrefillRequest, DecodeRequest, GenerationParameters,
};

/// gRPC 客户端封装
#[derive(Clone)]
pub struct GrpcClient {
    inner: Arc<parking_lot::Mutex<TextGenerationServiceClient<Channel>>>,
}

impl GrpcClient {
    /// 建立 gRPC 连接
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        tracing::info!("正在连接 gRPC 服务: {} ...", addr);
        
        // 重试连接
        let channel = loop {
            match Channel::from_shared(addr.to_string()) {
                Ok(endpoint) => {
                    match endpoint.connect().await {
                        Ok(ch) => break ch,
                        Err(e) => {
                            tracing::warn!("gRPC 连接失败，1秒后重试: {}", e);
                            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("无效地址: {}", e);
                    anyhow::bail!("无效的 gRPC 地址: {}", e);
                }
            }
        };

        let client = TextGenerationServiceClient::new(channel);
        tracing::info!("gRPC 连接成功");

        Ok(Self {
            inner: Arc::new(parking_lot::Mutex::new(client)),
        })
    }

    /// 获取模型信息
    pub async fn get_model_info(&self) -> anyhow::Result<ModelInfoResponse> {
        let mut client = self.inner.lock();
        let req = ModelInfoRequest {};
        let resp = client.health(req).await?;
        Ok(resp.into_inner())
    }

    /// 批量预填充
    pub async fn batch_prefill(
        &self,
        requests: Vec<PrefillRequest>,
        batch_id: i64,
    ) -> anyhow::Result<BatchPrefillResponse> {
        let mut client = self.inner.lock();
        let req = BatchPrefillRequest {
            batch_id,
            requests,
        };
        let resp = client.prefill(req).await?;
        Ok(resp.into_inner())
    }

    /// 批量解码
    pub async fn batch_decode(
        &self,
        requests: Vec<DecodeRequest>,
        batch_id: i64,
    ) -> anyhow::Result<BatchDecodeResponse> {
        let mut client = self.inner.lock();
        let req = BatchDecodeRequest {
            batch_id,
            requests,
        };
        let resp = client.decode(req).await?;
        Ok(resp.into_inner())
    }

    /// 清理 KV cache
    pub async fn clear_cache(&self, cache_handles: Vec<i64>) -> anyhow::Result<()> {
        let mut client = self.inner.lock();
        let req = ClearCacheRequest { cache_handles };
        let _resp = client.clear_cache(req).await?;
        Ok(())
    }
}
