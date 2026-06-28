# Light TGI — 轻量级 LLM 推理调度器设计文档

> **生产级 v4.0** | Continuous Batching + Prefix Sharing | 模型: Qwen2.5-1.5B-Instruct | 日期: 2026-06-28

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构总览](#2-架构总览)
3. [v3→v4: 从事件驱动到真正的 Continuous Batching](#3-v3v4-从事件驱动到真正的-continuous-batching)
4. [核心：Continuous Batching](#4-核心continuous-batching)
5. [Prefix Sharing (前缀共享)](#5-prefix-sharing-前缀共享)
6. [PagedAttention KV Cache](#6-pagedattention-kv-cache)
7. [请求生命周期](#7-请求生命周期)
8. [Protobuf 协议设计](#8-protobuf-协议设计)
9. [Rust Router 源码详解](#9-rust-router-源码详解)
10. [Python Model Server 源码详解](#10-python-model-server-源码详解)
11. [配置与调优](#11-配置与调优)
12. [运行指南](#12-运行指南)
13. [性能分析与扩展](#13-性能分析与扩展)

---

## 1. 项目概述

### 1.1 什么是 Light TGI

Light TGI 是 HuggingFace [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) 的轻量级生产版本。采用 **Rust 前端 + Python 后端** 的混合架构，实现了生产级的 Continuous Batching 和 Prefix Sharing。

### 1.2 核心特性 (v4)

| 特性 | 说明 |
|------|------|
| **Continuous Batching** | ★ 真正的 batch forward — 多个请求共享同一次 GPU 推理，吞吐量提升 N 倍 |
| **Prefix Sharing** | 共享前缀缓存 — 相同 system prompt 的请求复用 KV Cache blocks |
| **PagedAttention** | vLLM 风格分页 KV Cache — 零显存浪费，按需分配 block |
| **BatchStreamGenerate** | 双向流 gRPC — Rust 推送请求，Python 批量 forward，流式返回 token |
| **事件驱动架构** | EventBus + Actor 模型，完全异步非阻塞 |
| **真实模型推理** | 加载 HuggingFace Qwen2.5-1.5B-Instruct 进行真实推理 |
| **过载保护** | Semaphore 信号量限制并发，超出立即返回 429 |
| **流式输出** | SSE (Server-Sent Events) 实时推送生成的 token |
| **K8S 部署** | Dockerfile + Deployment + Service + HPA，生产就绪 |

### 1.3 与 TGI / vLLM 的对比

| 维度 | TGI (生产级) | vLLM | Light TGI v4 |
|------|-------------|------|-------------|
| 模型加载 | FlashAttention + TP | PagedAttention + CUDA kernels | HuggingFace + PagedAttention |
| KV Cache | PagedAttention / FlashInfer | PagedAttention (CUDA) | PagedAttention (PyTorch) |
| Continuous Batching | ★ 原生支持 | ★ 原生支持 | ★ 原生支持 (BatchScheduler) |
| Prefix Sharing | 支持 | 支持 (APC) | ★ 支持 (PrefixCache LRU) |
| 调度器 | Actor 模型 | Scheduler + BlockManager | BatchScheduler (Python) |
| gRPC 通信 | 双向流 | N/A | BatchStreamGenerate 双向流 |
| 量化 | GPT-Q, AWQ, FP8 | GPT-Q, AWQ, FP8 | float16 (GPU) |
| 代码量 | ~50K Rust + ~20K Python | ~40K Python | ~2.5K Rust + ~1.5K Python |

---

## 2. 架构总览

### 2.1 生产级架构 (v4)

```
┌──────────────────────────────────────────────────────────────────┐
│                      客户端 (HTTP Client)                         │
│                  POST /generate  {"inputs": "..."}                │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP / SSE
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              第一层: Rust Router (事件驱动 + 双向流)               │
│                                                                  │
│  ┌──────────┐     ┌─────────────────┐    ┌──────────────────┐   │
│  │ server.rs │────▶│   EventBus      │───▶│   Scheduler      │   │
│  │ HTTP入口  │发布 │ (全局消息中枢)   │订阅 │ (Continuous Batch)│   │
│  │ 验证/限流 │     │                 │    │                  │   │
│  └──────────┘     │ - NewRequest    │    │  BatchStreamGen  │   │
│       ▲           │ - TokenEvent    │    │  双向流 ──────────│───┼──┐
│       │ 订阅      └─────────────────┘    └──────────────────┘   │  │
│       │           TokenEvent ◀────────── BatchStreamResponse    │  │
│       │                                                         │  │
└───────┼─────────────────────────────────────────────────────────┘  │
        │ SSE Stream                                                 │ gRPC
        ▼                                                           │ Bidirectional
    Client (实时接收 token)                                          │ Stream
                                                                     │
┌──────────────────────────────────────────────────────────────────┐  │
│           第二层: Python Model Server (Batch Scheduler)          │◀─┘
│                                                                  │
│  ┌─────────────────┐    ┌────────────────┐    ┌──────────────┐  │
│  │ grpc_server.py  │───▶│ BatchScheduler │───▶│ ModelEngine  │  │
│  │ BatchStreamGen  │    │                │    │ Qwen2.5-1.5B │  │
│  │ PrefixCache RPC │    │ - pending queue│    │ PagedAttn    │  │
│  └─────────────────┘    │ - active pool  │    │ BlockPool    │  │
│                         │ - PrefixCache  │    └──────────────┘  │
│                         │ - step() loop  │                      │
│                         └────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么是 Rust + Python？

| 层 | 语言 | 原因 |
|----|------|------|
| **Router** | **Rust** | 无 GC 停顿、无 GIL、Tokio 异步高性能、极低内存占用 |
| **Model Server** | **Python** | PyTorch/Transformers 生态、HuggingFace 模型兼容 |

---

## 3. v3→v4: 从事件驱动到真正的 Continuous Batching

### 3.1 v3 的问题：假 Batch

```
v3 架构 (事件驱动, 但非真正 batch):
┌───────────────────────────────────────────────────────────────────┐
│  Scheduler                                                        │
│    │                                                              │
│    ├── Session Actor A ── StreamGenerate ──▶ Python (独立 prefill) │
│    ├── Session Actor B ── StreamGenerate ──▶ Python (独立 prefill) │
│    └── Session Actor C ── StreamGenerate ──▶ Python (独立 prefill) │
│                                                                   │
│  每个 Session 独立调用 gRPC, Python 端串行执行                      │
│  GPU forward 只能处理 1 个请求 → 吞吐量 = 单请求吞吐量              │
└───────────────────────────────────────────────────────────────────┘
```

**问题**：虽然 Rust 端是事件驱动、多 Session 并发，但每个 Session 各自调用 `StreamGenerate` RPC，Python 端**串行处理**每个请求的 prefill→decode 循环。GPU 一次只计算一个请求。

### 3.2 v4 解决方案：真正的 Continuous Batching

```
v4 架构 (BatchStreamGenerate 双向流):
┌───────────────────────────────────────────────────────────────────┐
│  Rust Router                                                      │
│    │                                                              │
│    │  BatchCommand { ADD_REQUEST [A, B, C] }                     │
│    ▼                                                              │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Python BatchScheduler                                │        │
│  │                                                       │        │
│  │  Step 1: batch_prefill([A, B, C])  → 1 次 GPU forward │        │
│  │  Step 2: batch_decode([A, B, C])   → 1 次 GPU forward │        │
│  │  Step 3: batch_decode([A, B, C])   → 1 次 GPU forward │        │
│  │  Step 4: 新增 D → batch_decode([A,B,C,D])             │        │
│  │  ...                                                  │        │
│  │  Step N: B 完成 → batch_decode([A, C, D])             │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                   │
│  一次 GPU forward 处理 N 个请求 → 吞吐量提升 N 倍                   │
└───────────────────────────────────────────────────────────────────┘
```

**关键变更**：

| | v3 (事件驱动) | v4 (Continuous Batching) |
|---|---|---|
| gRPC 接口 | `StreamGenerate` (单请求) | `BatchStreamGenerate` (双向流) |
| Session 管理 | Rust 端 spawn N 个 Session Actor | Python 端 BatchScheduler 统一管理 |
| GPU forward | 每次处理 1 个请求 | 每次处理 N 个请求 (batch) |
| 新请求加入 | 需要 Scheduler 协调 | 直接通过双向流发送 ADD_REQUEST |
| 吞吐量 | O(1) per step | O(N) per step |
| 延迟 | 低 (无排队) | 略高 (batch 等待) 但总吞吐大幅提升 |

---

## 4. 核心：Continuous Batching

### 4.1 什么是 Continuous Batching？

**传统 Static Batching 的问题**：

```
时间轴 ──────────────────────────────────────────────▶
请求A: [=====Prefill=====][D][D][D][D][D][D][D][D][D][D]
请求B: [==Prefill==][D][D][D]
请求C:        [=======等待=======][=====Prefill=====][D][D]...

GPU利用率: ████░░░░░░░░░░░░████████░░░░░░░░░░░░  (大量空闲)
```

**Continuous Batching 的解决方案**：

```
时间轴 ──────────────────────────────────────────────▶
请求A: [==Prefill==][D][D][D][D][D][D][D][D][D][D]  (完成)
请求B: [==Prefill==][D][D][D]                         (完成)
请求C:        [==Prefill==][D][D][D][D]...            (随时加入)
请求D:              [==Prefill==][D][D]...             (随时加入)

GPU利用率: ████████████████████████████████████████  (始终高利用率)
```

### 4.2 BatchScheduler 实现

**文件**: `model_server/batch_scheduler.py`

```
BatchScheduler 工作循环:

  loop:
    1. 检查 pending 队列 → 有则做 batch_prefill
       - 收集请求 → 调用 engine.batch_prefill()
       - 生成首 token → 加入 active decode pool
       - 存入 PrefixCache

    2. 检查 active pool → 有则做 batch_decode
       - 收集所有 active 请求的 last_token
       - 调用 engine.batch_decode() → 一次 GPU forward
       - 分发 token → 完成的请求移出 pool

    3. 输出结果 → 通过 gRPC stream 返回 Rust Router
```

**Prefill 与 Decode 混合执行**：

当前实现采用交替模式（prefill step → decode step），更高级的实现可以混合 prefill 和 decode 在同一次 forward 中（需要 FlashAttention 支持），但当前简化实现已能获得显著的批量收益。

### 4.3 Batch Decode 实现细节

```python
def batch_decode(self, request_ids, cache_handles, token_ids, params_list):
    """
    多个请求的 next_token 组成 batch

    关键步骤:
      1. 构建 batch input: (batch_size, 1)
      2. 合并 past_key_values: left-padding 对齐不同 seq_len
      3. 一次 model.forward(input_ids, past_key_values=merged_kv)
      4. 从 batch logits 中取每个请求的 logits
      5. 各自采样 + 更新 past_key_values
    """
```

### 4.4 性能收益

| 场景 | v3 吞吐 | v4 吞吐 | 提升 |
|------|---------|---------|------|
| 1 并发 | ~12 tok/s | ~12 tok/s | 1× |
| 4 并发 | ~12 tok/s | ~40 tok/s | 3.3× |
| 8 并发 | ~12 tok/s | ~70 tok/s | 5.8× |
| 16 并发 | ~12 tok/s | ~100 tok/s | 8.3× |

*(理论值，实际取决于模型大小和硬件)*

---

## 5. Prefix Sharing (前缀共享)

### 5.1 动机

许多 LLM 应用共享相同的 prompt 前缀：

- **客服机器人**: 所有请求共享相同的 system prompt
- **RAG 应用**: 共享检索到的 context
- **Few-shot prompting**: 共享示例前缀

传统方式下，每个请求都重新计算整个前缀的 KV Cache，造成大量冗余计算。

### 5.2 实现原理

```
PrefixCache (LRU):

  Key: 前缀文本 (最长匹配)
  Value: (cache_handle, token_count, block_ids)

流程:
  1. 新请求到达 → PrefixCache.lookup(prefix_text)
  2. 命中: 跳过前缀 prefill, 只处理 suffix
     - 复用共享的 BlockTable entries
     - 引用计数管理 (防止提前释放)
  3. 未命中: 正常 prefill → 完成后 store 到 PrefixCache
  4. LRU 淘汰: 缓存满时淘汰最久未使用的条目
```

### 5.3 BlockPool 引用共享

```
请求A: system_prompt = "You are a helpful assistant."
请求B: system_prompt = "You are a helpful assistant."

PrefixCache 命中 → B 的 BlockTable 前 6 个 block 指向与 A 相同的物理 block:

  BlockPool:
    [B0][B1][B2][B3][B4][B5][B6][B7][B8]...

  请求A BlockTable: [B0, B1, B2, B3, B4, B5, B10, B11]  ← 共享前缀
  请求B BlockTable: [B0, B1, B2, B3, B4, B5, B20, B21]  ← 共享前缀

  (B0-B5 被 A 和 B 共同引用, 通过引用计数管理)
```

### 5.4 PrefixCache gRPC 接口

```protobuf
rpc PrefixCacheLookup (PrefixCacheLookupRequest) returns (PrefixCacheLookupResponse);
rpc PrefixCacheStore (PrefixCacheStoreRequest) returns (PrefixCacheStoreResponse);
```

---

## 6. PagedAttention KV Cache

### 6.1 设计动机

传统方式预分配 `max_seq_len` 大小的连续 KV Cache：

```
传统 KV Cache:
  请求A: [████████████████░░░░░░░░░░░░░░░░]  (预分配 4096, 实际用 1200)
  请求B: [██████░░░░░░░░░░░░░░░░░░░░░░░░░░]  (预分配 4096, 实际用 400)

  浪费率: 70%+
```

PagedAttention 按需分配 block：

```
Paged KV Cache (block_size=16):
  BlockPool: [B0][B1][B2]...[B255]

  请求A (1200 tokens): 75 blocks → [B5, B12, B3, ...]  (精确分配)
  请求B (400 tokens):  25 blocks → [B7, B20, ...]       (精确分配)

  浪费率: 0%
```

### 6.2 数据结构

```python
# BlockPool: GPU 显存中的物理存储
# 形状: (num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim)
#       └─ block ─┘ └─ layer ┘ └K/V┘ └── head ──┘ └─ tokens ─┘

# BlockTable: 每个请求的逻辑→物理映射
# Request A: [physical_block_7, physical_block_3, physical_block_12, ...]

# PagedKVCache: 高层封装
# - 管理 BlockTable
# - 自动分配/释放 block
# - 同步 HF past_key_values (当前实现)
```

### 6.3 当前实现状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| BlockPool 存储 | ✅ 已实现 | GPU 显存中分配 block pool |
| BlockTable 映射 | ✅ 已实现 | 逻辑位置→物理 block 映射 |
| K/V 写入 | ✅ 已实现 | Prefill 后写入完整 K/V，Decode 时追加 |
| Block 分配/释放 | ✅ 已实现 | 按需分配，完成后归还 |
| 从 BlockPool 计算 Attention | ⚠️ 桥接模式 | 当前仍通过 HF past_key_values 做 attention |
| Prefix Sharing | ✅ 已实现 | PrefixCache + BlockTable 引用共享 |
| 自定义 CUDA Kernel | ❌ 未来 | 需要编写 CUDA kernel 直接从 BlockPool 计算 |

**当前采用桥接模式**：BlockPool 存储 K/V 用于 Prefix Sharing 和显存管理，实际 attention 计算仍使用 HuggingFace 的 `past_key_values`。这保证了正确性，同时 BlockPool 为 Prefix Sharing 提供了基础设施。后续可通过编写自定义 CUDA kernel 直接从 BlockPool 计算 attention，彻底替代 HF 的 past_key_values。

---

## 7. 请求生命周期

### 7.1 完整时间线 (v4)

```
Client          HTTP Server       EventBus       Scheduler        Python BatchScheduler
  │                 │                │               │                    │
  │─ POST /gen ────▶│                │               │                    │
  │                 │─ Tokenize ────────────────────────────────────────▶│
  │                 │◀── count ────────────────────────────────────────│
  │                 │─ validate     │               │                    │
  │                 │─ try_acquire  │               │                    │
  │                 │               │               │                    │
  │                 │─ publish ────▶│ NewRequest    │                    │
  │◀─ SSE stream ───│               │               │                    │
  │                 │               │── recv ──────▶│                    │
  │                 │               │               │─ BatchCommand ───▶│
  │                 │               │               │  ADD_REQUEST[A,B]  │
  │                 │               │               │                    │─ batch_prefill
  │                 │               │               │◀─ BatchStreamResp─│  token_A, token_B
  │                 │               │◀── TokenEvent ────────────────────│
  │◀─ SSE: tok_A ───│               │               │                    │
  │◀─ SSE: tok_B ───│               │               │                    │
  │                 │               │               │                    │─ batch_decode
  │                 │               │               │◀─ BatchStreamResp─│  token_A2, token_B2
  │                 │               │◀── TokenEvent ────────────────────│
  │◀─ SSE: tok_A2 ──│               │               │                    │
  │                 │               │               │         ...        │
  │                 │               │               │◀─ BatchStreamResp─│  token_A_last (EOS)
  │                 │               │◀── TokenEvent ────────────────────│
  │◀─ SSE: [DONE] ──│               │               │                    │
```

### 7.2 请求状态机

```
                    ┌──────────┐
        HTTP 请求 ─▶│ PENDING   │  等待 Scheduler flush
                    └────┬─────┘
                         │ BatchCommand(ADD_REQUEST)
                         ▼
                    ┌──────────┐
                    │PREFILLING│  Python 端 batch prefill
                    └────┬─────┘
                         │ 首 token 生成
                         ▼
                    ┌──────────┐
              ┌────▶│DECODING  │  Python 端 batch decode pool
              │     └────┬─────┘
              │          │ 每次 step 生成 1 token
              └──────────┘ (继续 decode)
                         │ EOS / max_tokens
                         ▼
                    ┌──────────┐
                    │ FINISHED │  清理 KV Cache + BlockTable
                    └──────────┘
```

---

## 8. Protobuf 协议设计

**文件**: `proto/generation.proto`

### 8.1 服务定义 (v4)

```protobuf
service TextGenerationService {
  rpc Health (ModelInfoRequest) returns (ModelInfoResponse);
  rpc Tokenize (TokenizeRequest) returns (TokenizeResponse);

  // ★ v4 核心: 双向流 Continuous Batching
  rpc BatchStreamGenerate (stream BatchCommand) returns (stream BatchStreamResponse);

  // ★ Prefix Sharing
  rpc PrefixCacheLookup (PrefixCacheLookupRequest) returns (PrefixCacheLookupResponse);
  rpc PrefixCacheStore (PrefixCacheStoreRequest) returns (PrefixCacheStoreResponse);

  // 单请求流式 (兼容/调试)
  rpc StreamGenerate (StreamGenerateRequest) returns (stream StreamGenerateResponse);

  // 批量接口 (保留兼容)
  rpc Prefill (BatchPrefillRequest) returns (BatchPrefillResponse);
  rpc Decode (BatchDecodeRequest) returns (BatchDecodeResponse);
  rpc ClearCache (ClearCacheRequest) returns (ClearCacheResponse);
}
```

### 8.2 BatchStreamGenerate 消息流

```
Rust → Python (BatchCommand):
  command_type: ADD_REQUEST
  new_requests: [StreamGenerateRequest, ...]

Python → Rust (BatchStreamResponse):
  tokens: [StreamGenerateResponse, ...]  ← 所有请求的本步 token
  active_requests: 5                      ← 当前活跃请求数
```

### 8.3 为什么双向流比单请求流更好？

| | v3 StreamGenerate (单请求) | v4 BatchStreamGenerate (双向流) |
|---|---|---|
| 连接数 | N 个请求 = N 个 stream | N 个请求 = 1 个 stream |
| GPU forward | 每次处理 1 个请求 | 每次处理 N 个请求 |
| 新请求加入 | 需要新开 stream | 通过已有 stream 发 ADD_REQUEST |
| 吞吐量 | O(1) | O(N) |
| 网络开销 | N 个连接 | 1 个连接 |

---

## 9. Rust Router 源码详解

### 9.1 项目结构 (v4)

```
router/
├── Cargo.toml          # 依赖管理 (新增 futures-util)
├── build.rs            # 编译 proto → Rust 代码
└── src/
    ├── main.rs         # 入口: 初始化 EventBus、Scheduler、HTTP Server
    ├── config.rs       # 配置管理 (环境变量)
    ├── server.rs       # HTTP Server (发布事件 + 订阅 token)
    ├── event_bus.rs    # 全局消息中枢
    ├── scheduler.rs    # ★ Scheduler (v4: BatchStreamGenerate 双向流)
    ├── session.rs      # Session Actor (v4: 保留兼容)
    ├── queue.rs        # 数据结构定义
    └── infer.rs        # ★ gRPC 客户端 (新增 BatchStreamGenerate)
```

### 9.2 Scheduler (v4) — 核心变更

**旧 (v3)**: 为每个请求 spawn 独立 Session Actor → 各自调用 `StreamGenerate` RPC
**新 (v4)**: 维护单条 `BatchStreamGenerate` 双向流 → 所有请求通过同一流交互

```rust
// scheduler.rs — v4
pub async fn run(mut self) {
    // ★ 双向流通道
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<BatchCommand>();
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    // spawn 双向流任务
    tokio::spawn(async move {
        grpc.batch_stream_generate(cmd_rx, token_tx).await
    });

    loop {
        tokio::select! {
            // 1. 新请求到达 → 积累到 pending
            Ok(event) = new_request_rx.recv() => { self.pending.push(event); }

            // 2. 定时 flush → 打包成 BatchCommand 发给 Python
            _ = ticker.tick() => { self.flush_pending(&cmd_tx).await; }

            // 3. 接收 Python 返回的 token → 分发到各 HTTP handler
            Some(batch_resp) = token_rx.recv() => { self.handle_batch_response(batch_resp); }
        }
    }
}
```

### 9.3 EventBus — 全局消息中枢

```rust
pub struct EventBus {
    new_request_tx: broadcast::Sender<NewRequestEvent>,     // HTTP → Scheduler
    token_tx: broadcast::Sender<QueueToken>,                // Scheduler → HTTP
    schedule_commands: Arc<DashMap<String, UnboundedSender>>, // P2P 指令
}
```

---

## 10. Python Model Server 源码详解

### 10.1 项目结构 (v4)

```
model_server/
├── grpc_server.py       # ★ gRPC 服务 (新增 BatchStreamGenerate + PrefixCache RPC)
├── model_engine.py      # 真实 HuggingFace 模型推理引擎 (PagedAttention + batch_decode)
├── batch_scheduler.py   # ★ Continuous Batching 调度器 (v4 新增)
├── paged_attention.py   # Paged KV Cache 实现 (BlockPool + BlockTable + PrefixCache)
├── generate_proto.py    # Proto 编译脚本
└── test_client.py       # gRPC 测试客户端
```

### 10.2 BatchScheduler — 核心调度器

```python
class BatchScheduler:
    """
    Continuous Batching 调度器

    核心数据结构:
      pending: deque          — 等待 prefill 的请求
      active: OrderedDict     — 正在 decode 的请求 (request_id → ActiveRequest)
      prefix_cache: PrefixCache — 前缀共享缓存 (LRU)

    核心方法:
      add_request()    — 添加新请求到 pending
      step()           — 执行一步调度 (prefill + decode)
      _prefill_step()  — 批量 prefill pending 请求
      _decode_step()   — 批量 decode 所有 active 请求
    """

    def step(self):
        results = []
        # 1. Prefill: pending → active
        results.extend(self._prefill_step())
        # 2. Decode: 所有 active 请求一起 forward
        if self.active:
            results.extend(self._decode_step())
        return results
```

### 10.3 ModelEngine — batch_decode 方法

```python
def batch_decode(self, request_ids, cache_handles, token_ids, params_list):
    """
    真正的 batch decode: 多个请求一次 model.forward()

    关键步骤:
      1. 构建 batch input: (batch_size, 1)
      2. 合并 past_key_values: left-padding 对齐
      3. model.forward(input_ids, past_key_values=merged_kv)
      4. 切分 batch logits → 各自采样
      5. 切分 batch past_key_values → 各自保存
    """
```

### 10.4 PrefixCache

```python
class PrefixCache:
    """
    LRU 前缀缓存

    lookup(prefix_text, min_match_tokens) → {cache_handle, token_count, block_ids}
    store(prefix_text, cache_handle, token_count, block_ids)

    使用 OrderedDict 实现 LRU 淘汰
    支持最长前缀匹配
    """
```

---

## 11. 配置与调优

### 11.1 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ROUTER_HOST` | `0.0.0.0` | HTTP 监听地址 |
| `ROUTER_PORT` | `3000` | HTTP 端口 |
| `MODEL_SERVER_HOST` | `127.0.0.1` | Python 服务地址 |
| `MODEL_SERVER_PORT` | `50051` | gRPC 端口 |
| `MAX_CONCURRENT_REQUESTS` | `64` | 最大并发请求数 |
| `MAX_BATCH_SIZE` | `8` | 单 batch 最大请求数 |
| `MAX_BATCH_PREFILL_TOKENS` | `4096` | 单 batch prefill token 上限 |
| `MAX_ACTIVE_REQUESTS` | `32` | 活跃 decode 请求上限 |
| `PREFIX_CACHE_SIZE` | `64` | PrefixCache 最大条目数 |
| `MAX_INPUT_LENGTH` | `2048` | 最大输入长度 |
| `MAX_TOTAL_TOKENS` | `4096` | 单请求最大总 token 数 |
| `MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace 模型 ID |
| `DEVICE` | `cpu` | 运行设备 (cpu/cuda) |
| `DTYPE` | `auto` | 数据类型 (auto/float16/bfloat16) |

### 11.2 调优建议

| 场景 | 推荐配置 |
|------|---------|
| **在线聊天 (低延迟)** | `MAX_BATCH_SIZE=4`, `MAX_CONCURRENT_REQUESTS=128` |
| **批量处理 (高吞吐)** | `MAX_BATCH_SIZE=16`, `MAX_BATCH_PREFILL_TOKENS=8192` |
| **客服机器人 (Prefix Sharing)** | `PREFIX_CACHE_SIZE=128`, 共享 system prompt |
| **CPU 部署 (小模型)** | `MODEL_ID=Qwen2.5-0.5B`, `DEVICE=cpu`, `MAX_BATCH_SIZE=4` |
| **GPU 部署 (大模型)** | `MODEL_ID=Qwen2.5-7B`, `DEVICE=cuda`, `DTYPE=float16` |

### 11.3 延迟分解

```
总延迟 = 排队时间 + Prefill 时间 + (Decode 时间 × 生成 token 数)

排队时间:  由并发数和 batch 等待策略决定
Prefill:   与 prompt 长度正相关 (O(n²) attention)
           batch prefill 会略增延迟, 但吞吐大幅提升
Decode:    每次 ~2-5ms per token (batch decode 下, N 个请求共享)
           单请求 vs batch 的 decode 延迟几乎相同 (batch 优势!)
```

---

## 12. 运行指南

### 12.1 环境准备

**Rust 环境**：
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# 安装 protoc
# Ubuntu: sudo apt install protobuf-compiler
# macOS: brew install protobuf
```

**Python 环境**：
```bash
cd light_tgi/model_server
pip install grpcio grpcio-tools protobuf torch transformers
```

### 12.2 编译 proto

```bash
# Python gRPC 代码
cd light_tgi/model_server
python generate_proto.py

# Rust gRPC 代码 (编译时自动, 通过 build.rs)
```

### 12.3 启动服务

**终端 1 — Python Model Server (v4)**：
```bash
cd light_tgi/model_server
python grpc_server.py --port 50051
# 输出: ★ BatchStreamGenerate + Prefix Sharing 已启用
```

**终端 2 — Rust Router (v4)**：
```bash
cd light_tgi/router
cargo run --release
# 输出: Light TGI Router v4 (Continuous Batching) 启动中
```

### 12.4 测试

```bash
# HTTP API 测试
python test_api.py

# 并发测试
python test_api.py --concurrent

# curl 测试
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is machine learning?", "parameters": {"max_new_tokens": 30}}'
```

---

## 13. 性能分析与扩展

### 13.1 关键指标

| 指标 | 说明 | 获取方式 |
|------|------|---------|
| 排队时间 | 请求在 pending 队列的等待时间 | `x-queue-time` 响应头 |
| Batch size | 当前 batch 中的请求数 | `active_requests` 字段 |
| 吞吐量 | 每秒生成的 token 数 | 总 tokens / 总时间 |
| Prefix cache hit rate | 前缀缓存命中率 | stats API |
| Block pool usage | KV cache block 使用率 | stats API |
| GPU 利用率 | GPU SM 使用率 | `nvidia-smi` |

### 13.2 扩展路线图

```
v4.0 (当前): Continuous Batching + Prefix Sharing + PagedAttention (桥接模式)
    │
    ├── v4.1: 自定义 CUDA Attention Kernel
    │         - 直接从 BlockPool 读取 K/V 计算 attention
    │         - 替代 HF past_key_values
    │         - FlashAttention 集成
    │
    ├── v4.2: 高级调度策略
    │         - Prefill/Decode 混合 batch (FlashAttention 支持)
    │         - 优先级调度
    │         - Chunked Prefill (长 prompt 分块)
    │
    ├── v4.3: 量化与优化
    │         - INT8/INT4 权重量化
    │         - KV Cache INT8 量化
    │         - Speculative Decoding
    │
    └── v4.4: 分布式
              - Tensor Parallel (多 GPU)
              - Pipeline Parallel
              - 多节点部署
```

### 13.3 K8S 部署架构

```
                        ┌──────────────┐
                        │  LoadBalancer │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │   Router Service    │
                    │   port: 80→3000     │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
        │ Router Pod  │ │ Router Pod  │ │ Router Pod  │  (HPA: 2-10)
        │ Rust/Axum   │ │ Rust/Axum   │ │ Rust/Axum   │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
               │               │               │
               └───────────────┼───────────────┘
                               │ gRPC (BatchStreamGenerate)
                    ┌──────────▼──────────┐
                    │ Model Server Service │
                    │   port: 50051        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Model Server Pod   │  (单副本, GPU)
                    │  Python BatchScheduler│
                    │  Qwen2.5-1.5B       │
                    └─────────────────────┘
```

### B. 参考资料

- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- [TGI Architecture](https://hugging-face.cn/docs/text-generation-inference/architecture)
- [vLLM: Continuous Batching](https://arxiv.org/abs/2308.09596)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Axum 框架](https://github.com/tokio-rs/axum)
- [Tonic gRPC](https://github.com/hyperium/tonic)
