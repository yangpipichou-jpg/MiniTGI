//! gRPC 推理客户端模块 (v4: Continuous Batching)
//!
//! 变更:
//!   - 新增 BatchStreamGenerate: 双向流, Rust 推送请求, Python 批量 forward
//!   - 新增 PrefixCacheLookup / PrefixCacheStore
//!   - 保留旧接口兼容

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
    BatchCommand, BatchStreamResponse,
    BatchCommandType,
    PrefixCacheLookupRequest,
};

/// gRPC 客户端 — 使用 Arc 而非 Mutex，因为 tonic 的 Channel/Client 内部已处理并发
/// 
/// 关键修复: 之前用 `Arc<Mutex<Client>>`，batch_stream_generate 持有 Mutex 期间
/// 其他调用 (如 tokenize) 会死锁。现在改为 Arc 直接 clone，每个调用获取独立的 client clone。
#[derive(Clone)]
pub struct GrpcClient {
    channel: Channel,
}

impl GrpcClient {
    /// 获取一个新的 client (每次调用都 clone，tonic client clone 是轻量的)
    fn client(&self) -> TextGenerationServiceClient<Channel> {
        TextGenerationServiceClient::new(self.channel.clone())
    }

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

        tracing::info!("gRPC 连接成功");
        Ok(Self { channel })
    }

    /// 获取模型信息
    pub async fn get_model_info(&self) -> anyhow::Result<ModelInfoResponse> {
        let mut client = self.client();
        let resp = client.health(ModelInfoRequest {}).await?;
        Ok(resp.into_inner())
    }

    /// Tokenize: 文本 → token 数
    pub async fn tokenize(&self, text: &str) -> anyhow::Result<usize> {
        let mut client = self.client();
        let req = TokenizeRequest { text: text.to_string() };
        let resp = client.tokenize(req).await?;
        Ok(resp.into_inner().token_count as usize)
    }

    /// ★ BatchStreamGenerate: 双向流 Continuous Batching (v4 核心)
    /// 
    /// 注意: 使用独立的 client clone，不持有全局锁。
    /// 这样其他调用 (tokenize, health) 可以并发执行。
    pub async fn batch_stream_generate(
        &self,
        cmd_rx: tokio::sync::mpsc::UnboundedReceiver<BatchCommand>,
        token_tx: tokio::sync::mpsc::UnboundedSender<BatchStreamResponse>,
    ) -> anyhow::Result<()> {
        use futures_util::StreamExt;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        // ★ 使用独立的 client clone，不阻塞其他调用
        let mut client = self.client();

        // Rust → Python stream
        let request_stream = UnboundedReceiverStream::new(cmd_rx);

        // 发起双向流调用
        let mut response_stream = client
            .batch_stream_generate(request_stream)
            .await?
            .into_inner();

        // 消费 Python → Rust stream, 转发到 token_tx
        while let Some(response) = response_stream.next().await {
            match response {
                Ok(batch_resp) => {
                    if token_tx.send(batch_resp).is_err() {
                        tracing::warn!("[gRPC] token_tx 已关闭, 停止接收");
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("[gRPC] BatchStreamGenerate 错误: {}", e);
                    return Err(anyhow::anyhow!("BatchStreamGenerate error: {}", e));
                }
            }
        }

        tracing::info!("[gRPC] BatchStreamGenerate 流结束");
        Ok(())
    }

    /// Prefix Cache 查找
    pub async fn prefix_cache_lookup(
        &self,
        prefix_text: &str,
        min_match_tokens: i32,
    ) -> anyhow::Result<(bool, i64, i32)> {
        let mut client = self.client();
        let req = PrefixCacheLookupRequest {
            prefix_text: prefix_text.to_string(),
            min_match_tokens,
        };
        let resp = client.prefix_cache_lookup(req).await?.into_inner();
        Ok((resp.found, resp.shared_cache_handle, resp.matched_tokens))
    }

    /// 批量预填充 (保留兼容)
    pub async fn batch_prefill(
        &self,
        requests: Vec<PrefillRequest>,
        batch_id: i64,
    ) -> anyhow::Result<BatchPrefillResponse> {
        let mut client = self.client();
        let req = BatchPrefillRequest { batch_id, requests };
        let resp = client.prefill(req).await?;
        Ok(resp.into_inner())
    }

    /// 批量解码 (保留兼容)
    pub async fn batch_decode(
        &self,
        requests: Vec<DecodeRequest>,
        batch_id: i64,
    ) -> anyhow::Result<BatchDecodeResponse> {
        let mut client = self.client();
        let req = BatchDecodeRequest { batch_id, requests };
        let resp = client.decode(req).await?;
        Ok(resp.into_inner())
    }

    /// 清理 KV cache
    pub async fn clear_cache(&self, cache_handles: Vec<i64>) -> anyhow::Result<()> {
        let mut client = self.client();
        let _resp = client.clear_cache(ClearCacheRequest { cache_handles }).await?;
        Ok(())
    }
}
