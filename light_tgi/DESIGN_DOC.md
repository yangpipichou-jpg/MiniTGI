# Light TGI — 轻量级 LLM 推理调度器设计文档

> **教学项目** | 参考 HuggingFace Text Generation Inference (TGI) 架构设计  
> 版本: v3.0 (事件驱动架构) | 模型: Qwen2.5-1.5B-Instruct | 日期: 2026-06-25

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构总览](#2-架构总览)
3. [架构升级：v2 → v3 事件驱动](#3-架构升级v2--v3-事件驱动)
4. [三层架构详解](#4-三层架构详解)
5. [核心：事件驱动的 Continuous Batching](#5-核心事件驱动的-continuous-batching)
6. [请求生命周期](#6-请求生命周期)
7. [Protobuf 协议设计](#7-protobuf-协议设计)
8. [Rust Router 源码详解](#8-rust-router-源码详解)
9. [Python Model Server 源码详解](#9-python-model-server-源码详解)
10. [配置与调优](#10-配置与调优)
11. [运行指南](#11-运行指南)
12. [性能分析与扩展](#12-性能分析与扩展)

---

## 1. 项目概述

### 1.1 什么是 Light TGI

Light TGI 是 HuggingFace [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) 的教学简化版本。它是一个**高性能 LLM 推理部署框架**，采用 **Rust 前端 + Python 后端** 的混合架构。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **事件驱动架构** | EventBus + Actor 模型，完全异步非阻塞 |
| **真实模型推理** | 加载 HuggingFace Qwen2.5-1.5B-Instruct 进行真实推理 |
| **Continuous Batching** | 动态批处理，新请求可随时加入正在运行的 batch |
| **Session Actor** | 每个请求独立生命周期管理，并发无阻塞 |
| **过载保护** | Semaphore 信号量限制并发，超出立即返回 429 |
| **流式输出** | SSE (Server-Sent Events) 实时推送生成的 token |
| **StreamGenerate RPC** | 异步流式 gRPC，Python 端内部控制完整生成流程 |
| **KV Cache 复用** | past_key_values 在 Prefill 和 Decode 间传递，避免重复计算 |
| **K8S 部署** | Dockerfile + Deployment + Service + HPA，生产就绪 |

### 1.3 与 TGI 的对比

| 维度 | TGI (生产级) | Light TGI v3 (事件驱动) |
|------|-------------|------------------------|
| 模型加载 | 真实权重 + FlashAttention + Tensor Parallel | 真实 HuggingFace 模型 (Qwen2.5-1.5B) |
| KV Cache | PagedAttention / FlashInfer | past_key_values 复用 |
| 量化 | GPT-Q, AWQ, FP8, ... | 支持 float16 (GPU) |
| 调度器 | Actor 模型 + 事件驱动 | ★ Actor 模型 + EventBus |
| Session 管理 | 每请求独立 Session | ★ Session Actor (tokio::spawn) |
| gRPC 通信 | 流式 + 批量 | ★ StreamGenerate RPC (异步流式) |
| Tokenization | Rust 端独立 tokenizer 线程 | gRPC Tokenize RPC (Python 端) |
| 部署 | Docker + K8S | Docker + K8S (含 HPA) |
| 代码量 | ~50K 行 Rust + ~20K 行 Python | ~2.5K 行 Rust + ~600 行 Python |

---

## 2. 架构总览

### 2.1 事件驱动架构

```
┌──────────────────────────────────────────────────────────────┐
│                      客户端 (HTTP Client)                     │
│                  POST /generate  {"inputs": "..."}            │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP / SSE
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  第一层: Rust Router (事件驱动)               │
│                                                              │
│  ┌──────────┐     ┌─────────────────┐    ┌──────────────┐   │
│  │ server.rs │────▶│   EventBus      │───▶│ Scheduler    │   │
│  │ HTTP入口  │发布 │ (全局消息中枢)   │订阅 │ Actor        │   │
│  │ 验证/限流 │     │                 │    │ 组Batch/协调  │   │
│  └──────────┘     │ - NewRequest    │    └──────┬───────┘   │
│       ▲           │ - TokenEvent    │           │ spawn     │
│       │ 订阅      │ - SessionEvent  │           ▼           │
│       │           │ - ScheduleCmd   │    ┌──────────────┐   │
│       │           └─────────────────┘    │ Session      │   │
│       │                                  │ Actor × N    │   │
│       │                                  │ 独立生命周期  │   │
│       │                                  └──────┬───────┘   │
│       │                                         │ gRPC      │
│       │         ◀─── TokenEvent ────────────────┘           │
└───────┼─────────────────────────────────────────────────────┘
        │ SSE Stream
        ▼
    Client (实时接收 token)
```

### 2.2 为什么升级到事件驱动？

| 维度 | v2 (同步阻塞) | v3 (事件驱动) |
|------|-------------|--------------|
| 调度器 | 阻塞在 grpc_client.prefill() | Actor 非阻塞, tokio::select! 多路复用 |
| Session | 无, batch 共享生命周期 | 每请求独立 Session Actor |
| 通信 | mpsc channel (点对点) | EventBus broadcast (多对多) |
| 并发 | 同一 batch 串行 decode | 多个 Session 并发 decode |
| 可扩展 | 单调度器瓶颈 | 多 Actor 天然分布式友好 |

### 2.3 为什么是 Rust + Python？

| 层 | 语言 | 原因 |
|----|------|------|
| **Router** | **Rust** | 无 GC 停顿、无 GIL、零成本抽象、Tokio 异步高性能 |
| **Model Server** | **Python** | PyTorch/Transformers 生态、HuggingFace 模型兼容 |

TGI 的设计哲学：**让 Rust 做它擅长的（高并发网络IO），让 Python 做它擅长的（ML推理）**。

---

## 3. 架构升级：v2 → v3 事件驱动

### 3.1 旧架构的问题 (v2 同步阻塞)

```
HTTP Request
  │
  ▼
server.rs (Axum) ──阻塞等待──▶ scheduler.rs ──阻塞等待──▶ grpc_client.rs ──阻塞等待──▶ Python
  │                                                                                        │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
                                                                         返回
```

**三大问题**：
1. **调度器阻塞**: `scheduler.run()` 在 `run_prefill()` 和 `run_decode()` 中阻塞等待 gRPC 响应
2. **Batch 串行**: 当前 batch 未完成时，无法处理新请求
3. **无并发 Session**: 所有请求共享 batch 生命周期，无法独立管理

### 3.2 新架构 (v3 事件驱动)

```
HTTP Request ──► [EventBus] ──► Scheduler Actor ◄───┐
                      ▲              │  ▲            │
                      │              ▼  │            │
                      │        Spawn Session Actor  │
                      │              │  │           │
                      │              ▼  │           │
                      └─── [Events] ◄───┴───────────┘
                                      │
                                      ▼
                             gRPC Client (Async Stream)
                                      │
                                      ▼
                               Python Worker
```

**核心组件**：

| 组件 | 职责 | 文件 |
|------|------|------|
| **EventBus** | 全局消息中枢，解耦所有组件通信 | `event_bus.rs` |
| **Scheduler Actor** | 订阅事件、组 Batch、协调 Session | `scheduler.rs` |
| **Session Actor** | 每请求独立生命周期、调用 gRPC | `session.rs` |
| **HTTP Server** | 验证、过载保护、发布事件、SSE 流 | `server.rs` |

### 3.3 EventBus 设计

```
┌─────────────────────────────────────────────────────────────┐
│                        EventBus                              │
│                                                             │
│  NewRequest Channel (broadcast)                             │
│    HTTP Server ──publish──▶ ◇ ──subscribe──▶ Scheduler     │
│                                                             │
│  SessionEvent Channel (broadcast)                           │
│    Session Actor ──publish──▶ ◇ ──subscribe──▶ Scheduler   │
│                                                             │
│  TokenEvent Channel (broadcast)                             │
│    Session Actor ──publish──▶ ◇ ──subscribe──▶ HTTP Server │
│                                                             │
│  ScheduleCommand (point-to-point, DashMap)                  │
│    Scheduler ──send──▶ DashMap[request_id] ──recv──▶ Session│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**设计要点**：
- `broadcast` 用于一对多事件 (NewRequest, TokenEvent, SessionEvent)
- `DashMap + mpsc` 用于点对点指令 (ScheduleCommand → 特定 Session)
- 每个 channel 容量 1024，防止慢消费者阻塞生产者
- EventBus 是 `Clone` 的，所有组件持有同一实例的引用

### 3.4 Session Actor 状态机

```
                    ┌──────────┐
    spawn ────────▶│ CREATED   │
                    └────┬─────┘
                         │ StartPrefill
                         ▼
                    ┌──────────┐
                    │PREFILLING│──────────▶ ERROR
                    └────┬─────┘
                         │ PrefillDone
                         ▼
                    ┌──────────┐
               ┌───▶│DECODING  │──────────▶ ERROR
               │    └────┬─────┘
               │         │ DecodeDone(!finished)
               └─────────┘
                         │ DecodeDone(finished)
                         ▼
                    ┌──────────┐
                    │ FINISHED │
                    └──────────┘
```

### 3.5 关键差异对比

| | v2 (同步) | v3 (事件驱动) |
|---|---|---|
| 调度器运行方式 | `loop { wait; prefetch; decode_loop }` | `tokio::select! { events }` |
| 请求生命周期 | 绑定到 batch | 独立 Session Actor |
| gRPC 调用 | Prefill + N×Decode 独立调用 | StreamGenerate 一次流式调用 |
| SSE 数据源 | 独占 mpsc channel | EventBus broadcast 按 request_id 过滤 |
| 并发能力 | 单 batch 串行 | 多 Session 并发 |

---

## 4. 三层架构详解

### 4.1 第一层：Rust Router (事件驱动)

**文件**: `router/src/server.rs`, `router/src/main.rs`

Router 是一个基于 [Axum](https://github.com/tokio-rs/axum) 框架的 HTTP 服务器，职责：

1. **接收 HTTP 请求** — `POST /generate`
2. **请求验证** — 检查输入长度、参数合法性 (gRPC Tokenize)
3. **过载保护** — 使用 `Semaphore::try_acquire()` 限制并发
4. **★ 发布事件** — 发布 `NewRequestEvent` 到 EventBus (不再入队)
5. **★ 订阅 Token** — 从 EventBus 订阅 token 事件, 按 request_id 过滤
6. **返回 SSE 流** — 立即返回流式响应，不等待推理完成

关键代码路径：

```rust
// server.rs - generate_handler() (v3 事件驱动)
async fn generate_handler(State(state): State<AppState>, Json(payload): Json<GenerateRequest>) {
    // 1. 验证
    let (params, input_text) = validate_request(&payload, &state).await?;
    
    // 2. 过载保护
    let permit = state.semaphore.clone().try_acquire_owned()?;
    
    // 3. ★ 订阅 EventBus token 事件
    let mut token_rx = state.event_bus.subscribe_token();
    
    // 4. ★ 发布新请求事件 (Scheduler 会收到)
    state.event_bus.publish_new_request(NewRequestEvent {
        request_id, input_text, params, ...
    });
    
    // 5. 返回 SSE 流 (从 EventBus 过滤自己的 token)
    Sse::new(build_sse_stream(request_id, token_rx, permit, queue_time))
}
```

**SSE 流过滤机制**：
```rust
fn build_sse_stream(request_id, mut token_rx, ...) -> impl Stream {
    async_stream::stream! {
        loop {
            match token_rx.recv().await {
                Ok(token) => {
                    if token.request_id != request_id { continue; }  // ★ 过滤
                    yield Event::default().data(...);
                    if token.is_finished { break; }
                }
                ...
            }
        }
    }
}
```

### 4.2 第二层：EventBus (全局消息中枢)

**文件**: `router/src/event_bus.rs`

EventBus 是 v3 架构的核心，用 4 种 channel 实现完全解耦：

| Channel | 类型 | 生产者 | 消费者 | 用途 |
|---------|------|--------|--------|------|
| `new_request_tx` | broadcast | server.rs | scheduler.rs | 新请求通知 |
| `session_tx` | broadcast | session.rs | scheduler.rs | Session 状态变更 |
| `token_tx` | broadcast | session.rs | server.rs | Token 生成通知 |
| `schedule_commands` | DashMap + mpsc | scheduler.rs | session.rs | 一对一调度指令 |

### 4.3 第三层：Python Model Server

**文件**: `model_server/grpc_server.py`, `model_server/model_engine.py`

v3 新增 **StreamGenerate RPC**：

```protobuf
service TextGenerationService {
  // ★ 异步流式生成 (v3 核心)
  rpc StreamGenerate (StreamGenerateRequest) returns (stream StreamGenerateResponse);
  
  // 保留批量接口 (兼容)
  rpc Prefill (BatchPrefillRequest) returns (BatchPrefillResponse);
  rpc Decode (BatchDecodeRequest) returns (BatchDecodeResponse);
}
```

**StreamGenerate 流程**：
```
Session Actor                    Python gRPC Server
    │                                   │
    │── StreamGenerateRequest ─────────▶│
    │   (input_text + params)           │── tokenize
    │                                   │── model.forward (prefill)
    │◀── StreamGenerateResponse ───────│   (step=0, token_1, cache_handle)
    │                                   │
    │                                   │── model.forward (decode, past_key_values)
    │◀── StreamGenerateResponse ───────│   (step=1, token_2)
    │                                   │
    │                                   │── ... 循环 ...
    │                                   │
    │◀── StreamGenerateResponse ───────│   (step=N, finish_reason=EOS)
    │                                   │── clear_cache
```

---

## 5. 核心：事件驱动的 Continuous Batching

**文件**: `router/src/scheduler.rs`, `router/src/session.rs`, `router/src/event_bus.rs`

这是 v3 最核心的模块，实现了**事件驱动的 Continuous Batching**。

### 5.1 Scheduler Actor 主循环

```rust
// scheduler.rs - SchedulerActor::run() (v3 事件驱动)
pub async fn run(mut self) {
    let mut new_request_rx = self.event_bus.subscribe_new_request();
    let mut session_event_rx = self.event_bus.subscribe_session_event();
    let mut ticker = tokio::time::interval(Duration::from_millis(50));

    loop {
        tokio::select! {
            // ★ 事件1: 新请求到达
            Ok(event) = new_request_rx.recv() => {
                self.pending.push(event);  // 非阻塞!
            }

            // ★ 事件2: Session 状态变更
            Ok(event) = session_event_rx.recv() => {
                self.handle_session_event(event).await;  // 非阻塞!
            }

            // ★ 事件3: 定时器触发组 batch
            _ = ticker.tick() => {
                if !self.pending.is_empty() {
                    self.try_schedule_batch().await;  // 非阻塞!
                }
            }
        }
    }
}
```

**关键差异**：v2 的 `run()` 在 `run_prefill()` 和 `run_decode()` 中阻塞；v3 使用 `tokio::select!` 同时监听多种事件，**从不阻塞**。

### 5.2 预算驱动的组 Batch 策略

`try_schedule_batch()` 与 v2 的 `collect_batch()` 逻辑相同，但改为非阻塞：

```rust
async fn try_schedule_batch(&mut self) {
    let batch_size = self.pending.len().min(self.config.max_batch_size);
    let to_schedule: Vec<_> = self.pending.drain(..batch_size).collect();

    for req in to_schedule {
        // ★ 为每个请求创建独立调度通道
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        self.event_bus.register_schedule_channel(req.request_id.clone(), cmd_tx);

        // ★ spawn Session Actor (独立生命周期)
        let session = Session::new(req, ..., cmd_rx);
        tokio::spawn(async move { session.run().await });
    }

    // ★ 向所有 Session 发送 StartPrefill (非阻塞!)
    for request_id in &request_ids {
        self.event_bus.send_schedule_command(request_id, ScheduleCommand::StartPrefill { ... });
    }
}
```

### 5.3 什么是 Continuous Batching？

**传统 Static Batching 的问题**：

```
时间轴 ──────────────────────────────────────────────▶
请求A: [=====Prefill=====][D][D][D][D][D][D][D][D][D][D]  (生成10 token)
请求B: [==Prefill==][D][D][D]                                (生成3 token)
请求C:        [=======等待=======][=====Prefill=====][D][D]...

传统批处理：A和B完成后，C才能开始 → 大量 GPU 空闲时间
```

**Continuous Batching 的解决方案**：

```
时间轴 ──────────────────────────────────────────────▶
请求A: [=====Prefill=====][D][D][D][D][D][D][D][D][D][D]  (完成，移出)
请求B: [==Prefill==][D][D][D]                              (完成，移出)
请求C:        [=====Prefill=====][D][D][D]...               (随时加入)
请求D:              [==Prefill==][D][D][D][D]...            (随时加入)

GPU利用率: ████████████████████████████████████████  (始终保持高利用率)
```

### 5.4 waiting_served_ratio 权衡

```
                   低延迟偏好 ←──────────→ 高吞吐偏好
                   (ratio=1.5)              (ratio=0.3)
                         │                      │
  行为:  立即组小batch    │        等待形成大batch  │
  延迟:  ★★★★★ (很低)    │    ★★☆ (较高)          │
  吞吐:  ★★☆ (较低)      │    ★★★★★ (很高)        │
  适用:  在线聊天         │    离线批量处理         │
```

### 5.5 Prefill 与 Decode 的区别

```
┌─────────────────────────────────────────────────────────────────┐
│                        Prefill (预填充)                          │
├─────────────────────────────────────────────────────────────────┤
│ 输入:  完整的 prompt (n 个 tokens)                               │
│ 输出:  第 1 个生成的 token + KV Cache handle                     │
│ 特点:  计算密集型 (O(n²) attention)                              │
│ 延迟:  较高 (与 prompt 长度正相关)                                │
│ 产出:  KV Cache (所有层的 key-value 矩阵)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (KV Cache 被复用)
┌─────────────────────────────────────────────────────────────────┐
│                         Decode (解码)                            │
├─────────────────────────────────────────────────────────────────┤
│ 输入:  上一步生成的 token + KV Cache handle                      │
│ 输出:  下一个 token                                              │
│ 特点:  内存密集型 (O(1) 新计算，O(n) KV Cache 访问)              │
│ 延迟:  很低 (每步 ~几ms)                                         │
│ 循环:  直到遇到 EOS 或达到 max_new_tokens                        │
└─────────────────────────────────────────────────────────────────┘
```

**为什么分开？**
- Prefill 和 Decode 的计算特性完全不同
- 分开后可以在同一次 GPU forward 中混合执行（FlashAttention 支持）
- Decode 复用 Prefill 的 KV Cache，避免重复计算

---

## 6. 请求生命周期 (v3)

### 6.1 完整时间线 (事件驱动)

```
Client          HTTP Server       EventBus       Scheduler        Session Actor    Python
  │                 │                │               │                │              │
  │─ POST /gen ────▶│                │               │                │              │
  │                 │─ Tokenize ──────────────────────────────────────────────────▶│
  │                 │◀── count ──────────────────────────────────────────────────│
  │                 │─ validate     │               │                │              │
  │                 │─ try_acquire  │               │                │              │
  │                 │               │               │                │              │
  │                 │─ publish ────▶│ NewRequest    │                │              │
  │◀─ SSE stream ──│               │               │                │              │
  │                 │               │── recv ──────▶│                │              │
  │                 │               │               │─ spawn ──────▶│              │
  │                 │               │               │─ StartPrefill─▶│              │
  │                 │               │               │                │─ StreamGen ─▶│
  │                 │               │◀── TokenEvent ────────────────│◀─ token_1 ──│
  │◀─ SSE: tok1 ───│               │               │                │              │
  │                 │               │               │◀─ PrefillDone──│              │
  │                 │               │               │─ StartDecode ─▶│              │
  │                 │               │◀── TokenEvent ────────────────│◀─ token_2 ──│
  │◀─ SSE: tok2 ───│               │               │                │              │
  │                 │               │               │    ... (重复)   │              │
  │                 │               │◀── TokenEvent ────────────────│◀─ EOS ─────│
  │◀─ SSE: [DONE] ─│               │               │                │              │
  │                 │               │               │◀─ DecodeDone───│              │
```

### 6.2 请求状态机

```
                    ┌──────────┐
        HTTP 请求 ─▶│ CREATED   │  Session Actor spawn
                    └────┬─────┘
                         │ StartPrefill
                         ▼
                    ┌──────────┐
                    │PREFILLING│  预填充 (处理 prompt)
                    └────┬─────┘
                         │ PrefillDone → Scheduler 发 StartDecode
                         ▼
                    ┌──────────┐
              ┌────▶│DECODING  │  逐个生成 token
              │     └────┬─────┘
              │          │ DecodeDone(!finished) → Scheduler 再发 StartDecode
              └──────────┘
                         │ DecodeDone(finished)
                         ▼
                    ┌──────────┐
                    │ FINISHED │  完成，清理资源
                    └──────────┘
```

### 6.3 错误处理路径

```
验证失败 (输入为空/过长/参数非法)
    → 422 Unprocessable Entity

过载 (并发数超限)
    → 429 Too Many Requests (Semaphore::try_acquire 失败)

Session Prefill 错误
    → Session 发布 SessionEvent::Error 到 EventBus
    → Scheduler 收到后清理该 Session 记录
    → Session 发布 TokenEvent(is_finished=true, finish_reason="error")

Session Decode 错误
    → 同上流程
```

---

## 7. Protobuf 协议设计

**文件**: `proto/generation.proto`

### 7.1 服务定义 (v3)

```protobuf
service TextGenerationService {
  rpc Health(ModelInfoRequest) returns (ModelInfoResponse);
  rpc Tokenize(TokenizeRequest) returns (TokenizeResponse);
  
  // ★ v3 核心: 异步流式生成
  rpc StreamGenerate (StreamGenerateRequest) returns (stream StreamGenerateResponse);
  
  // 批量接口 (保留兼容)
  rpc Prefill (BatchPrefillRequest) returns (BatchPrefillResponse);
  rpc Decode (BatchDecodeRequest) returns (BatchDecodeResponse);
  rpc ClearCache (ClearCacheRequest) returns (ClearCacheResponse);
}
```

**v3 关键变更**：
- ★ 新增 `StreamGenerate` RPC — 服务端流式，Python 端内部控制 Prefill→Decode 循环
- Session 发送一次请求，Python 持续 yield token 直到完成
- 替代 v2 的 Prefill + N×Decode 多次 RPC 调用

### 7.2 关键消息类型 (v3)

```protobuf
// ★ StreamGenerate: 一次请求, 流式响应
message StreamGenerateRequest {
  string request_id = 1;
  string input_text = 2;
  GenerationParameters params = 3;
}

message StreamGenerateResponse {
  string request_id = 1;
  Token generated_token = 2;
  FinishReason finish_reason = 3;
  int32 step = 4;                    // 当前步数 (prefill=0)
  int32 duration_ms = 5;
  int64 cache_handle = 6;            // prefill 步返回
}
```

### 7.3 为什么 StreamGenerate 比 Prefill+Decode 更好？

| | v2 Prefill+Decode | v3 StreamGenerate |
|---|---|---|
| RPC 调用次数 | 1 + N (N=生成token数) | 1 |
| 状态管理 | Rust 端管理循环 | Python 端内部控制 |
| 延迟 | 每次 Decode 都有 RTT | 流式推送, 低延迟 |
| Session 复杂度 | 需要协调 Prefill/Decode 指令 | 发一次请求, 消费 stream |

### 6.4 代码生成

- **Rust 端**: `build.rs` 使用 `tonic-build` 在编译时从 proto 生成
- **Python 端**: `generate_proto.py` 使用 `grpc_tools.protoc` 生成

---

## 8. Rust Router 源码详解

### 8.1 项目结构 (v3)

```
router/
├── Cargo.toml          # 依赖管理 (新增 async-stream, futures)
├── build.rs            # 编译 proto → Rust 代码
└── src/
    ├── main.rs         # 入口：初始化 EventBus、Scheduler、HTTP Server
    ├── config.rs       # 配置管理 (环境变量)
    ├── server.rs       # ★ HTTP Server (发布事件 + 订阅 token)
    ├── event_bus.rs    # ★ 全局消息中枢 (v3 新增)
    ├── scheduler.rs    # ★ Scheduler Actor (事件驱动)
    ├── session.rs      # ★ Session Actor (v3 新增)
    ├── queue.rs        # 数据结构定义 (保留)
    └── infer.rs        # gRPC 客户端封装
```

### 8.2 main.rs — 启动流程 (v3)

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 初始化
    let config = RouterConfig::from_env()?;
    let grpc_client = GrpcClient::connect(&grpc_addr).await?;

    // 2. ★ 创建 EventBus (全局消息中枢)
    let event_bus = EventBus::new();

    // 3. ★ 创建 Scheduler Actor
    let scheduler = SchedulerActor::new(config.clone(), grpc_client.clone(), event_bus.clone());

    // 4. ★ spawn Scheduler Actor
    tokio::spawn(async move { scheduler.run().await });

    // 5. 启动 HTTP Server (通过 event_bus 通信)
    let app_state = server::AppState { event_bus, semaphore, config, grpc_client };
    axum::serve(listener, app).await
}
```

### 8.3 event_bus.rs — 全局消息中枢

```rust
pub struct EventBus {
    new_request_tx: broadcast::Sender<NewRequestEvent>,     // HTTP → Scheduler
    session_tx: broadcast::Sender<SessionEvent>,            // Session → Scheduler
    token_tx: broadcast::Sender<QueueToken>,                // Session → HTTP
    schedule_commands: Arc<DashMap<String, UnboundedSender>>, // Scheduler → Session (P2P)
}
```

**为什么 broadcast 而不是 mpsc？**
- 多个 HTTP handler 同时订阅 token 事件 (每个 handler 过滤自己的 request_id)
- Scheduler 需要接收所有 Session 事件
- broadcast 天然支持多对多

**为什么 DashMap + mpsc 做点对点？**
- ScheduleCommand 是定向指令 (Scheduler → 特定 Session)
- 不能用 broadcast (会发给所有 Session)
- DashMap[request_id] 精确路由

### 8.4 server.rs — HTTP 处理 (v3)

**核心变更**：
- 不再入队到 mpsc channel
- 发布 NewRequestEvent 到 EventBus
- SSE 流从 EventBus 订阅 token 事件，按 request_id 过滤

```rust
// SSE 流过滤逻辑
async_stream::stream! {
    loop {
        match token_rx.recv().await {
            Ok(token) => {
                if token.request_id != request_id { continue; }  // ★ 关键过滤
                yield Event::default().data(...);
                if token.is_finished { break; }
            }
            Err(Lagged(n)) => continue,  // 丢帧恢复
            Err(Closed) => break,
        }
    }
}
```

### 8.5 scheduler.rs — Scheduler Actor (v3)

**核心变更**：
- 从 `BatchingScheduler::run()` 的阻塞循环 → `SchedulerActor::run()` 的事件驱动 `tokio::select!`
- 不再直接调用 grpc_client
- 改为 spawn Session Actor 并通过 EventBus 协调

### 8.6 session.rs — Session Actor (v3 新增)

```rust
pub struct Session {
    request_id: String,
    input_text: String,
    cache_handle: i64,
    generated_count: u32,
    grpc_client: GrpcClient,
    event_bus: EventBus,
    cmd_rx: UnboundedReceiver<ScheduleCommand>,  // 接收 Scheduler 指令
    state: SessionState,
}

impl Session {
    pub async fn run(mut self) {
        while self.state != Finished && self.state != Errored {
            match self.cmd_rx.recv().await {
                Some(StartPrefill { .. }) => self.do_prefill().await,
                Some(StartDecode { .. }) => self.do_decode().await,
                Some(Abort { .. }) => { /* 清理 */ break; }
                None => break,
            }
        }
    }
}
```

**Session 的生命周期完全独立**：
- spawn 后立即开始等待 Scheduler 指令
- 通过 EventBus 发布 token 和状态变更
- 完成后自动清理 (从 DashMap 注销)

---

## 9. Python Model Server 源码详解 (v3)

### 9.1 项目结构

```
model_server/
├── pyproject.toml        # Python 项目配置
├── generate_proto.py     # 编译 proto → Python 代码
├── grpc_server.py        # ★ gRPC 服务实现 (新增 StreamGenerate RPC)
├── model_engine.py       # 真实 HuggingFace 模型推理引擎
└── test_client.py        # gRPC 测试客户端
```

### 9.2 grpc_server.py — StreamGenerate RPC (v3 核心)

```python
def StreamGenerate(self, request, context):
    """流式生成: 接收一次请求, yield token 直到完成"""
    request_id = request.request_id
    
    # Step 1: Prefill
    cache_handle, tokens, _ = self.engine.prefill(...)
    yield StreamGenerateResponse(
        request_id=request_id,
        generated_token=Token(id=token["id"], text=token["text"]),
        step=0,
        cache_handle=cache_handle,
    )
    
    # Step 2: Decode 循环
    step = 1
    while True:
        token_dict, finish_reason, _ = self.engine.decode(...)
        if token_dict is None:
            break
        
        yield StreamGenerateResponse(
            request_id=request_id,
            generated_token=Token(...),
            finish_reason=reason_map.get(finish_reason),
            step=step,
        )
        
        if finish_reason:
            self.engine.clear_cache([cache_handle])
            return
        step += 1
```

**StreamGenerate 的优势**：
- Python 端内部控制 Prefill→Decode 循环，减少 RPC 往返
- Session 只需一次 gRPC 调用，消费 stream 即可
- 自然支持流式推送，延迟更低

### 9.3 model_engine.py — 推理引擎 (与 v2 相同)

`model_engine.py` 的核心实现与 v2 保持一致，包括：
- HuggingFace 模型加载 (`AutoModelForCausalLM`)
- Prefill/Decode 的 `past_key_values` KV Cache 复用
- Temperature/TopK/TopP 采样
- Left-padding 批量 Prefill

详细代码参见 v2 文档或源码文件。

---

## 10. 配置与调优

### 10.1 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ROUTER_HOST` | `0.0.0.0` | HTTP 监听地址 |
| `ROUTER_PORT` | `3000` | HTTP 端口 |
| `MODEL_SERVER_HOST` | `127.0.0.1` | Python 服务地址 |
| `MODEL_SERVER_PORT` | `50051` | gRPC 端口 |
| `MAX_CONCURRENT_REQUESTS` | `64` | 最大并发请求数 |
| `MAX_BATCH_SIZE` | `8` | 单 batch 最大请求数 (真实模型不宜过大) |
| `MAX_BATCH_PREFILL_TOKENS` | `2048` | 单 batch prefill token 上限 |
| `MAX_BATCH_TOTAL_TOKENS` | `8192` | 单 batch 总 token 上限 |
| `MAX_WAITING_TOKENS` | `12` | 触发组 batch 的等待请求数阈值 |
| `WAITING_SERVED_RATIO` | `1.2` | 延迟/吞吐平衡参数 |
| `MAX_INPUT_LENGTH` | `2048` | 最大输入长度 |
| `MAX_TOTAL_TOKENS` | `4096` | 单请求最大总 token 数 |
| `MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace 模型 ID |
| `DEVICE` | `cpu` | 运行设备 (cpu/cuda) |
| `DTYPE` | `auto` | 数据类型 (auto/float16/bfloat16) |

### 10.2 调优建议

| 场景 | 推荐配置 |
|------|---------|
| **在线聊天 (低延迟)** | `WAITING_SERVED_RATIO=1.5`, `MAX_BATCH_SIZE=2`, `MAX_CONCURRENT_REQUESTS=128` |
| **批量处理 (高吞吐)** | `WAITING_SERVED_RATIO=0.3`, `MAX_BATCH_SIZE=8`, `MAX_BATCH_PREFILL_TOKENS=4096` |
| **CPU 部署 (小模型)** | `MODEL_ID=Qwen2.5-1.5B`, `DEVICE=cpu`, `MAX_BATCH_SIZE=4` |
| **GPU 部署 (大模型)** | `MODEL_ID=Qwen2.5-7B`, `DEVICE=cuda`, `DTYPE=float16`, `MAX_BATCH_SIZE=16` |

### 10.3 延迟分解

```
总延迟 = 排队时间 + Prefill 时间 + (Decode 时间 × 生成 token 数)

排队时间:  由并发数和 waiting_served_ratio 决定
Prefill:   与 prompt 长度正相关 (O(n²) attention)
Decode:    每次 ~2-5ms，与生成 token 数线性相关
```

---

## 11. 运行指南

### 11.1 环境准备

**Rust 环境**：
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 protoc (protobuf 编译器)
# Ubuntu/Debian:
sudo apt install protobuf-compiler
# macOS:
brew install protobuf
# Windows:
# 下载 https://github.com/protocolbuffers/protobuf/releases
```

**Python 环境**：
```bash
cd light_tgi/model_server
pip install grpcio grpcio-tools protobuf torch
```

### 11.2 编译 proto

```bash
# 生成 Python gRPC 代码
cd light_tgi/model_server
python generate_proto.py
```

### 11.3 启动服务

**终端 1 — 启动 Python Model Server (v3)**：
```bash
cd light_tgi/model_server
python grpc_server.py --port 50051
# 输出: ★ StreamGenerate RPC 已启用 (异步流式)
```

**终端 2 — 编译并启动 Rust Router (v3)**：
```bash
cd light_tgi/router
cargo run --release
# 输出: Light TGI Router v3 (事件驱动架构) 启动中
```

### 11.4 测试

**测试 Python gRPC Server**：
```bash
cd light_tgi/model_server
python test_client.py
```

**测试 HTTP API**：
```bash
cd light_tgi
python test_api.py

# 并发测试
python test_api.py --concurrent
```

**使用 curl**：
```bash
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is machine learning?", "parameters": {"max_new_tokens": 30}}'
```

---

## 12. 性能分析与扩展

### 12.1 关键指标

| 指标 | 说明 | 获取方式 |
|------|------|---------|
| 排队时间 | 请求在队列中的等待时间 | `x-queue-time` 响应头 |
| Prefill 时间 | 处理 prompt 的耗时 | `prefill_duration_ms` 字段 |
| Decode 时间 | 每步生成 token 的耗时 | `decode_duration_ms` 字段 |
| 吞吐量 | 每秒生成的 token 数 | 总 tokens / 总时间 |
| 并发数 | 当前活跃请求数 | Semaphore available_permits |

### 12.2 扩展点

1. **PagedAttention**：将 KV Cache 分页管理，减少内存碎片
2. **Prefix Caching**：缓存共享的 prompt 前缀，避免重复 prefill
3. **Speculative Decoding**：用小模型"猜测"多个 token，大模型验证
4. **Tensor Parallel**：多 GPU 分片推理
5. **量化支持**：INT8/INT4 权重压缩

### 12.3 K8S 部署架构

```
                        ┌──────────────┐
                        │  LoadBalancer │  (对外暴露 HTTP)
                        │   / Ingress   │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │   Router Service    │  (ClusterIP/LoadBalancer)
                    │   port: 80→3000     │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
        │ Router Pod  │ │ Router Pod  │ │ Router Pod  │  (HPA: 2-10)
        │ Rust/Axum   │ │ Rust/Axum   │ │ Rust/Axum   │
        │ CPU: 100m   │ │ CPU: 100m   │ │ CPU: 100m   │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
               │               │               │
               └───────────────┼───────────────┘
                               │ gRPC
                    ┌──────────▼──────────┐
                    │ Model Server Service │  (ClusterIP)
                    │   port: 50051        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Model Server Pod   │  (单副本, GPU 可选)
                    │  Python/gRPC        │
                    │  Qwen2.5-1.5B       │
                    │  Mem: 4-8Gi         │
                    └─────────────────────┘
```

**关键 K8S 配置**:

| 组件 | 副本数 | 资源 | 说明 |
|------|--------|------|------|
| Router | 2-10 (HPA) | 64Mi-256Mi mem | Rust 极低资源消耗 |
| Model Server | 1 | 4Gi-8Gi mem | 模型加载开销大，单副本 |
| ConfigMap | — | — | 统一管理配置 |
| HPA | — | CPU 70% 触发 | Router 自动伸缩 |

### 12.4 从教学版到生产版

```
Light TGI v3 (事件驱动)                →    TGI (生产级)
─────────────────────────────────────────────
EventBus + Actor 模型               →    Ray / 分布式 Actor
Session per request                 →    Ray Serve Deployment
StreamGenerate RPC                  →    gRPC bidirectional stream
真实 HuggingFace 模型推理           →    FlashAttention 2 + vLLM
past_key_values 复用               →    PagedAttention (vLLM)
CPU / 单 GPU                        →    Tensor Parallel 多 GPU
K8S Deployment + HPA                →    + Service Mesh + Canary
无监控                              →    Prometheus + Grafana
HTTP + SSE                          →    + WebSocket + gRPC-Web
```

### B. 参考资料

- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- [TGI Architecture](https://hugging-face.cn/docs/text-generation-inference/architecture)
- [Continuous Batching 论文](https://arxiv.org/abs/2308.09596) (vLLM)
- [Axum 框架](https://github.com/tokio-rs/axum)
- [Tonic gRPC](https://github.com/hyperium/tonic)
