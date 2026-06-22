# Light TGI — 轻量级 LLM 推理调度器

> 教学项目 | 参考 [HuggingFace TGI](https://github.com/huggingface/text-generation-inference) 架构设计

Light TGI 是 Text Generation Inference (TGI) 的教学简化版本。它实现了 TGI 的核心架构：**Rust Router + Python Model Server + gRPC**，完整保留了 Continuous Batching 调度器的设计思想。

## 快速开始

### 1. 启动 Python Model Server

```bash
cd model_server
pip install grpcio grpcio-tools protobuf torch
python generate_proto.py
python grpc_server.py --port 50051
```

### 2. 启动 Rust Router

```bash
cd router
cargo run --release
```

### 3. 发送请求

```bash
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is machine learning?", "parameters": {"max_new_tokens": 30}}'
```

## 项目结构

```
light_tgi/
├── proto/generation.proto       # Protobuf 协议定义
├── router/                      # Rust HTTP Router + 调度器
│   └── src/
│       ├── main.rs              # 入口
│       ├── server.rs            # HTTP Server (Axum)
│       ├── queue.rs             # 请求队列 (mpsc + Notify)
│       ├── scheduler.rs         # 批处理调度器 (核心)
│       ├── infer.rs             # gRPC 客户端
│       └── config.rs            # 配置管理
├── model_server/                # Python gRPC 模型服务
│   ├── grpc_server.py           # gRPC 服务实现
│   ├── model_engine.py          # 推理引擎
│   └── test_client.py           # gRPC 测试
├── test_api.py                  # HTTP API 测试
└── DESIGN_DOC.md                # 详细设计文档
```

## 架构概览

```
Client (HTTP/SSE)
    │
    ▼
┌─────────────────────────────┐
│  Rust Router                │
│  ┌───────┐ ┌──────┐ ┌─────┐│
│  │Server │→│Queue │→│Sched│││
│  │ 验证   │ │无界通道│ │组B  │││
│  │ 限流   │ │通知  │ │atch │││
│  └───────┘ └──────┘ └──┬──┘││
└────────────────────────┼───┘
                         │ gRPC
                         ▼
┌─────────────────────────────┐
│  Python Model Server        │
│  ┌──────────┐ ┌───────────┐│
│  │gRPC Svr │→│ModelEngine│││
│  │          │ │Prefill    │││
│  │          │ │Decode     │││
│  └──────────┘ └───────────┘││
└─────────────────────────────┘
```

## 核心特性

- **Continuous Batching**: 动态批处理，新请求随时加入正在运行的 batch
- **过载保护**: Semaphore 信号量限制并发，超出返回 429
- **流式输出**: SSE (Server-Sent Events) 实时推送 token
- **预算驱动调度**: 基于 token 预算的智能组 batch
- **KV Cache 管理**: Prefill 产生的 KV Cache 被 Decode 复用

## 详细文档

请阅读 [DESIGN_DOC.md](./DESIGN_DOC.md) 获取完整的设计文档，包含：

- 架构详解
- Continuous Batching 算法原理
- 请求生命周期
- 源码逐模块详解
- 配置调优指南
- 性能分析

## 参考资料

- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- [TGI Architecture Docs](https://hugging-face.cn/docs/text-generation-inference/architecture)
- [vLLM: Continuous Batching](https://arxiv.org/abs/2308.09596)
