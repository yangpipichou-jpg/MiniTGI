# Light TGI — 轻量级 LLM 推理调度器

> **生产级 v4.0** | Continuous Batching + Prefix Sharing | 模型: **Qwen2.5-1.5B-Instruct** | Python: `F:\ProgramData\anaconda3\python.exe`

Light TGI 是 Text Generation Inference (TGI) 的轻量级生产版本，采用 **Rust Router + Python Model Server + gRPC 双向流** 架构。v4 实现了真正的 Continuous Batching 和 Prefix Sharing。

---

## 快速开始

### 环境要求

- **Python**: `F:\ProgramData\anaconda3\python.exe` (>=3.10)
- **Rust**: 最新稳定版 (https://rustup.rs)
- **protoc**: protobuf 编译器
- **模型**: 首次运行自动从 HuggingFace 下载 (~3GB)

### 启动

**终端 1 — Python Model Server**:

```bash
start_model_server.bat
```

> 首次运行会下载 Qwen2.5-1.5B-Instruct 模型 (~3GB)，请耐心等待。

**终端 2 — Rust Router**:

```bash
start_router.bat
```

### 测试

```bash
# HTTP API 测试
F:\ProgramData\anaconda3\python.exe test_api.py

# 并发测试
F:\ProgramData\anaconda3\python.exe test_api.py --concurrent

# curl 测试
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is machine learning?", "parameters": {"max_new_tokens": 30}}'
```

---

## 项目结构

```
light_tgi/
├── proto/generation.proto       # gRPC 协议 (v4: BatchStreamGenerate + PrefixCache)
├── router/                      # Rust HTTP Router + Scheduler
│   └── src/
│       ├── main.rs              # 入口
│       ├── server.rs            # HTTP Server (Axum) + Tokenize 验证
│       ├── scheduler.rs         # ★ Continuous Batching Scheduler
│       ├── session.rs           # Session Actor (保留兼容)
│       ├── queue.rs             # 数据结构
│       ├── infer.rs             # gRPC 客户端 (BatchStreamGenerate)
│       ├── event_bus.rs         # 全局消息中枢
│       └── config.rs            # 配置管理
├── model_server/                # Python gRPC 模型服务
│   ├── grpc_server.py           # ★ gRPC 服务 (BatchStreamGenerate + PrefixCache)
│   ├── batch_scheduler.py       # ★ Continuous Batching 调度器
│   ├── model_engine.py          # ★ 真实 HuggingFace 模型推理引擎 (PagedAttention)
│   ├── paged_attention.py       # ★ Paged KV Cache (BlockPool + BlockTable)
│   ├── generate_proto.py        # Proto 编译脚本
│   └── test_client.py           # gRPC 测试
├── k8s/                         # Kubernetes 部署
├── test_api.py                  # HTTP API 测试
├── start_model_server.bat       # 一键启动 Model Server
├── start_router.bat             # 一键编译启动 Router
├── DESIGN_DOC.md                # ★ 详细设计文档
└── README.md                    # 本文档
```

---

## 架构概览

```
Client (HTTP/SSE)
    │  POST /generate {"inputs": "What is AI?"}
    ▼
┌──────────────────────────────────────┐
│  Rust Router (HTTP Server)           │
│  ┌────────┐ ┌──────────┐ ┌────────┐ │
│  │ Server │→│ EventBus │→│Schedlr │ │
│  │ 验证    │ │          │ │Batch   │ │
│  │ 限流    │ │ Token    │ │Stream  │ │
│  └────────┘ └──────────┘ └───┬────┘ │
└───────────────────────────────┼──────┘
                                │ gRPC Bidirectional Stream
                                ▼
┌──────────────────────────────────────┐
│  Python Model Server                 │
│  ┌────────────┐ ┌──────────────────┐ │
│  │BatchSchedlr│→│  ModelEngine     │ │
│  │Continuous  │ │  Qwen2.5-1.5B    │ │
│  │Batching    │ │  PagedAttention  │ │
│  │PrefixCache │ │  BlockPool       │ │
│  └────────────┘ └──────────────────┘ │
└──────────────────────────────────────┘
```

## 核心特性

- **Continuous Batching**: 多个请求共享同一次 GPU forward，吞吐量提升 N 倍
- **Prefix Sharing**: 相同前缀的请求复用 KV Cache blocks
- **PagedAttention**: vLLM 风格分页 KV Cache，零显存浪费
- **BatchStreamGenerate**: 双向流 gRPC，Python 端批量调度
- **真实模型**: 加载 Qwen2.5-1.5B-Instruct 进行真实推理
- **过载保护**: Semaphore 信号量限制并发，超出返回 429
- **流式输出**: SSE 实时推送 token
- **K8S 就绪**: Dockerfile + Deployment + Service + HPA

## 支持的模型

| 模型 | 参数 | 推荐场景 |
|------|------|---------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | 最轻量, CPU 可用 |
| **`Qwen/Qwen2.5-1.5B-Instruct`** | **1.5B** | **默认推荐** |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | 中等性能 |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | GPU 推荐 |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | GPU 必须 |
| `google/gemma-2-2b-it` | 2B | Google 模型 |

## K8S 部署

```bash
# CPU 部署
kubectl apply -f k8s/deployment.yaml

# GPU 部署 (需 NVIDIA Device Plugin)
kubectl apply -f k8s/gpu-deployment.yaml

# 查看状态
kubectl -n light-tgi get pods,svc,hpa
```

## 详细文档

请阅读 [DESIGN_DOC.md](./DESIGN_DOC.md) 获取完整的设计文档。

## 参考资料

- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- [vLLM: Continuous Batching](https://arxiv.org/abs/2308.09596)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
