# Light TGI — 轻量级 LLM 推理调度器

> 教学项目 | 参考 [HuggingFace TGI](https://github.com/huggingface/text-generation-inference) 架构设计  
> **v3.0 事件驱动架构** | 模型: **Qwen2.5-1.5B-Instruct** | Python: `F:\ProgramData\anaconda3\python.exe`

Light TGI 是 Text Generation Inference (TGI) 的教学版本，采用 **Rust Router + Python Model Server + gRPC** 架构。v3 升级为**事件驱动架构** (EventBus + Actor 模型)，实现了非阻塞的 Continuous Batching 调度器。

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
> 模型加载完成后会显示: `模型加载完成: 1.54B 参数`

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
├── proto/generation.proto       # gRPC 协议 (v2: 真实模型版本)
├── router/                      # Rust HTTP Router + 调度器
│   └── src/
│       ├── main.rs              # 入口
│       ├── server.rs            # HTTP Server (Axum) + Tokenize 验证
│       ├── queue.rs             # 请求队列 (原始文本)
│       ├── scheduler.rs         # Continuous Batching 调度器 (核心)
│       ├── infer.rs             # gRPC 客户端 (含 Tokenize RPC)
│       └── config.rs            # 配置管理
├── model_server/                # Python gRPC 模型服务
│   ├── grpc_server.py           # gRPC 服务实现
│   ├── model_engine.py          # ★ 真实 HuggingFace 模型推理引擎
│   ├── generate_proto.py        # Proto 编译脚本
│   └── test_client.py           # gRPC 测试
├── k8s/                         # Kubernetes 部署
│   ├── Dockerfile.router        # Rust Router 镜像
│   ├── Dockerfile.model_server  # Python Model Server 镜像
│   ├── deployment.yaml          # K8S 部署清单 (CPU)
│   └── gpu-deployment.yaml      # K8S 部署清单 (GPU)
├── test_api.py                  # HTTP API 测试
├── start_model_server.bat       # 一键启动 Model Server
├── start_router.bat             # 一键编译启动 Router
├── DESIGN_DOC.md                # 详细设计文档
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
│  ┌────────┐ ┌────────┐ ┌──────────┐ │
│  │ Server │→│ Queue  │→│ Scheduler│ │
│  │ 验证    │ │ 原始文本│ │ 组Batch  │ │
│  │ 限流    │ │ 通知   │ │ Prefill  │ │
│  │ Tokenize│ │        │ │ Decode   │ │
│  └────────┘ └────────┘ └────┬─────┘ │
└──────────────────────────────┼───────┘
                               │ gRPC
                               ▼
┌──────────────────────────────────────┐
│  Python Model Server (gRPC)          │
│  ┌────────────┐ ┌──────────────────┐ │
│  │gRPC Server │→│  ModelEngine     │ │
│  │            │ │  Qwen2.5-1.5B    │ │
│  │            │ │  Tokenizer       │ │
│  │            │ │  model.forward() │ │
│  │            │ │  past_key_values │ │
│  └────────────┘ └──────────────────┘ │
└──────────────────────────────────────┘
```

## 核心特性

- **真实模型**: 加载 Qwen2.5-1.5B-Instruct 进行真实推理
- **Continuous Batching**: 动态批处理，最大化 GPU/CPU 利用率
- **过载保护**: Semaphore 信号量限制并发，超出返回 429
- **流式输出**: SSE 实时推送 token
- **预算驱动调度**: 基于 token 预算的智能组 batch
- **KV Cache 复用**: past_key_values 在 Prefill 和 Decode 间传递
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
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [vLLM: Continuous Batching](https://arxiv.org/abs/2308.09596)
