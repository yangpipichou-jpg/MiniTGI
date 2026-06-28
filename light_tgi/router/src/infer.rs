//! gRPC 推理客户端模块 (真实模型版本)
//!
//! 变更:
//!   - 新增 tokenize() 方法，供 Router 做输入长度验证
//!   - PrefillRequest 携带 input_text 而非 input_ids

use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Channel;

pub mod pb {
    tonic::include_proto!("light_tgi");
}

use pb::{
    text_generation_service_client::TextGenerationServiceClient,
    BatchPrefillRequest, BatchPrefillResponse,
    BatchDecodeRequest, BatchDecodeResponse,
    ClearCacheRequest,
    ModelInfoRequest, ModelInfoResponse,
    TokenizeRequest,
    PrefillRequest, DecodeRequest,
};

#[derive(Clone)]
pub struct GrpcClient {
    inner: Arc<Mutex<TextGenerationServiceClient<Channel>>>,
}

impl GrpcClient {
    /// 建立 gRPC 连接 (带重试)
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        tracing::info!("正在连接 gRPC 服务: {} ...", addr);

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
                Err(e) => anyhow::bail!("无效的 gRPC 地址: {}", e),
            }
        };

        let client = TextGenerationServiceClient::new(channel);
        tracing::info!("gRPC 连接成功");
        Ok(Self { inner: Arc::new(Mutex::new(client)) })
    }

    /// 获取模型信息
    pub async fn get_model_info(&self) -> anyhow::Result<ModelInfoResponse> {
        let mut client = self.inner.lock().await;
        let resp = client.health(ModelInfoRequest {}).await?;
        Ok(resp.into_inner())
    }

    /// ★ Tokenize: 文本 → token 数 (供 Router 做长度验证)
    pub async fn tokenize(&self, text: &str) -> anyhow::Result<usize> {
        let mut client = self.inner.lock().await;
        let req = TokenizeRequest { text: text.to_string() };
        let resp = client.tokenize(req).await?;
        Ok(resp.into_inner().token_count as usize)
    }

    /// 批量预填充
    pub async fn batch_prefill(
        &self,
        requests: Vec<PrefillRequest>,
        batch_id: i64,
    ) -> anyhow::Result<BatchPrefillResponse> {
        let mut client = self.inner.lock().await;
        let req = BatchPrefillRequest { batch_id, requests };
        let resp = client.prefill(req).await?;
        Ok(resp.into_inner())
    }

    /// 批量解码
    pub async fn batch_decode(
        &self,
        requests: Vec<DecodeRequest>,
        batch_id: i64,
    ) -> anyhow::Result<BatchDecodeResponse> {
        let mut client = self.inner.lock().await;
        let req = BatchDecodeRequest { batch_id, requests };
        let resp = client.decode(req).await?;
        Ok(resp.into_inner())
    }

    /// 清理 KV cache
    pub async fn clear_cache(&self, cache_handles: Vec<i64>) -> anyhow::Result<()> {
        let mut client = self.inner.lock().await;
        let _resp = client.clear_cache(ClearCacheRequest { cache_handles }).await?;
        Ok(())
    }
}
