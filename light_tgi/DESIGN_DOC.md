# Light TGI — 轻量级 LLM 推理调度器设计文档

> **教学项目** | 参考 HuggingFace Text Generation Inference (TGI) 架构设计  
> 作者：AI 教学助手 | 日期：2026-06-21

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构总览](#2-架构总览)
3. [三层架构详解](#3-三层架构详解)
4. [核心：Continuous Batching 调度器](#4-核心continuous-batching-调度器)
5. [请求生命周期](#5-请求生命周期)
6. [Protobuf 协议设计](#6-protobuf-协议设计)
7. [Rust Router 源码详解](#7-rust-router-源码详解)
8. [Python Model Server 源码详解](#8-python-model-server-源码详解)
9. [配置与调优](#9-配置与调优)
10. [运行指南](#10-运行指南)
11. [性能分析](#11-性能分析)

---

## 1. 项目概述

### 1.1 什么是 Light TGI

Light TGI 是 HuggingFace [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) 的教学简化版本。它是一个**高性能 LLM 推理部署框架**，采用 **Rust 前端 + Python 后端** 的混合架构。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **Continuous Batching** | 动态批处理，新请求可随时加入正在运行的 batch |
| **过载保护** | Semaphore 信号量限制并发，超出立即返回 429 |
| **流式输出** | SSE (Server-Sent Events) 实时推送生成的 token |
| **gRPC 通信** | Rust ↔ Python 通过 Protobuf 定义的 gRPC 高效通信 |
| **KV Cache 管理** | Prefill 生成的 KV Cache 被 Decode 复用 |
| **预算驱动调度** | 基于 token 预算动态组 batch，平衡延迟与吞吐量 |

### 1.3 与 TGI 的对比

| 维度 | TGI (生产级) | Light TGI (教学版) |
|------|-------------|-------------------|
| 模型加载 | 真实权重 + FlashAttention + Tensor Parallel | 模拟 token 生成 |
| KV Cache | PagedAttention / FlashInfer | 简化模拟 |
| 量化 | GPT-Q, AWQ, FP8, ... | 无 |
| 调度器 | 完整实现 | 核心算法保留 |
| 架构 | 完全相同 | 完全相同 |
| 代码量 | ~50K 行 Rust + ~20K 行 Python | ~1.5K 行 Rust + ~500 行 Python |

---

## 2. 架构总览

### 2.1 三层架构

```
┌──────────────────────────────────────────────────────────────┐
│                      客户端 (HTTP Client)                     │
│                  POST /generate  {"inputs": "..."}            │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP / SSE
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  第一层: Rust Router (HTTP Server)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ server.rs │  │ queue.rs │  │scheduler │  │  infer.rs    │ │
│  │ HTTP入口  │─▶│ 请求队列  │─▶│  调度器   │─▶│ gRPC Client  │ │
│  │ 验证/限流 │  │ 通知机制  │  │ 组Batch  │  │ 连接管理     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────┬───────┘ │
└───────────────────────────────────────────────────┼──────────┘
                                                    │ gRPC
                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│              第二层: Python Model Server (gRPC Server)        │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │  grpc_server.py   │  │       model_engine.py            │  │
│  │  gRPC 服务实现    │─▶│   Prefill / Decode / KV Cache   │  │
│  │  请求路由/转发    │  │   模拟 LLM 推理                  │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 为什么是 Rust + Python？

| 层 | 语言 | 原因 |
|----|------|------|
| **Router** | **Rust** | 无 GC 停顿、无 GIL、零成本抽象、Tokio 异步高性能 |
| **Model Server** | **Python** | PyTorch/Transformers 生态、HuggingFace 模型兼容 |

TGI 的设计哲学：**让 Rust 做它擅长的（高并发网络IO），让 Python 做它擅长的（ML推理）**。

### 2.3 数据流向

```
Client HTTP Request
    │
    ▼
[server.rs] ──验证──▶ [queue.rs] ──入队──▶ [scheduler.rs]
    │                                            │
    │ (立即返回 SSE)                              │ (组 Batch)
    ▼                                            ▼
Stream<Token> ◀── response_tx ◀────────── [grpc_client.prefill()]
                                                  │
                                                  │ gRPC
                                                  ▼
                                         [Python gRPC Server]
                                                  │
                                                  ▼
                                         [ModelEngine.prefill()]
                                         [ModelEngine.decode()]  ← 循环
```

---

## 3. 三层架构详解

### 3.1 第一层：Rust Router

**文件**: `router/src/server.rs`, `router/src/main.rs`

Router 是一个基于 [Axum](https://github.com/tokio-rs/axum) 框架的 HTTP 服务器，职责：

1. **接收 HTTP 请求** — `POST /generate`
2. **请求验证** — 检查输入长度、参数合法性
3. **过载保护** — 使用 `Semaphore::try_acquire()` 限制并发
4. **入队** — 将请求封装为 `QueueEntry` 放入 `RequestQueue`
5. **返回 SSE 流** — 立即返回流式响应，不等待推理完成

关键代码路径：

```rust
// server.rs - generate_handler()
async fn generate_handler(State(state): State<AppState>, Json(payload): Json<GenerateRequest>) {
    // 1. 验证
    let (input_ids, params) = validate_request(&payload, &state.config)?;
    
    // 2. 过载保护
    let permit = state.semaphore.clone().try_acquire_owned()
        .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, ...))?;
    
    // 3. 创建响应通道
    let (response_tx, response_rx) = mpsc::unbounded_channel();
    
    // 4. 入队 + 唤醒后台
    state.queue.enqueue(QueueEntry { ... });
    
    // 5. 返回 SSE 流
    Sse::new(build_sse_stream(response_rx, permit))
}
```

**过载保护机制**：
- 使用 `tokio::sync::Semaphore` 控制最大并发请求数
- `try_acquire()` 不等待，超出限制立即返回 `429 Too Many Requests`
- 这是**非阻塞**的保护机制，避免 HTTP 线程被阻塞

### 3.2 第二层：Request Queue

**文件**: `router/src/queue.rs`

队列是 Router 前台和后台的桥梁：

```
                     ┌──────────────────┐
  HTTP Thread ──────▶│  mpsc::Unbounded  │──────▶ Batching Task
  (server.rs)   send │     Channel       │  recv  (scheduler.rs)
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │    Notify         │  唤醒等待中的后台任务
                     │  (通知机制)       │
                     └──────────────────┘
```

**设计要点**：

| 组件 | 类型 | 作用 |
|------|------|------|
| `tx/rx` | `mpsc::unbounded_channel()` | 无界通道，HTTP 线程永不阻塞 |
| `notify` | `tokio::sync::Notify` | 一对多通知，新请求到达时唤醒后台 |
| `waiting_tokens` | `AtomicUsize` | 原子计数器，追踪队列中等待的 token 总数 |

**为什么用无界通道？**
- HTTP 线程不能阻塞（会导致请求超时）
- 后台任务通过 `waiting_served_ratio` 控制消费速率
- 如果队列积压，过载保护的 Semaphore 会限制新请求进入

### 3.3 第三层：Python Model Server

**文件**: `model_server/grpc_server.py`, `model_server/model_engine.py`

Python 端是一个 gRPC 服务器，实现 `TextGenerationService`：

```protobuf
service TextGenerationService {
  rpc Health(ModelInfoRequest) returns (ModelInfoResponse);
  rpc Prefill(BatchPrefillRequest) returns (BatchPrefillResponse);
  rpc Decode(BatchDecodeRequest) returns (BatchDecodeResponse);
  rpc ClearCache(ClearCacheRequest) returns (ClearCacheResponse);
}
```

**ModelEngine** 是推理核心，管理：
- KV Cache 的分配/查询/释放
- Prefill 计算（处理 prompt，生成首 token）
- Decode 计算（基于 KV Cache 逐个生成 token）

---

## 4. 核心：Continuous Batching 调度器

**文件**: `router/src/scheduler.rs`

这是整个项目最核心的模块，实现了 Continuous Batching（连续批处理）算法。

### 4.1 什么是 Continuous Batching？

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

### 4.2 调度器主循环

```rust
// scheduler.rs - BatchingScheduler::run()
pub async fn run(&self) {
    let mut receiver = self.queue.take_receiver();
    loop {
        // Step 1: 等待新请求通知
        self.queue.wait_for_notify().await;

        // Step 2: 从队列收集请求，按预算组 batch
        let batch_entries = self.collect_batch(&mut receiver).await;
        if batch_entries.is_empty() { continue; }

        // Step 3: Prefill 阶段
        self.run_prefill(batch_id, &mut requests).await?;

        // Step 4: Decode 循环
        loop {
            let active = requests.iter().filter(|r| !r.is_finished);
            if active.count() == 0 { break; }
            self.run_decode(batch_id, &mut requests).await?;
        }

        // Step 5: 清理 KV Cache
        self.grpc_client.clear_cache(cache_handles).await;
    }
}
```

### 4.3 预算驱动的组 Batch 策略

`collect_batch()` 方法基于三个约束组装 batch：

```rust
async fn collect_batch(&self, receiver) -> Vec<QueueEntry> {
    let mut entries = Vec::new();
    let mut token_count = 0;

    // 非阻塞收集请求
    while let Ok(entry) = receiver.try_recv() {
        // 约束 1: batch 大小限制
        if entries.len() >= self.config.max_batch_size { break; }
        
        // 约束 2: prefill token 预算
        if token_count + entry.input_ids.len() > self.config.max_batch_prefill_tokens { 
            break; 
        }
        
        token_count += entry.input_ids.len();
        entries.push(entry);
    }

    // 约束 3: waiting_served_ratio 决定是否等待更多请求
    let waiting_tokens = self.queue.waiting_tokens();
    let ratio = waiting_tokens as f64 / token_count as f64;
    if ratio < self.config.waiting_served_ratio {
        // 等待更多请求组成更大的 batch
        tokio::time::sleep(Duration::from_millis(5)).await;
        // 再尝试收集一次
    }

    entries
}
```

**三个预算约束**：

| 约束 | 参数 | 作用 |
|------|------|------|
| Batch 大小 | `max_batch_size` | 限制同时处理的请求数 |
| Prefill Token 预算 | `max_batch_prefill_tokens` | 限制单次 prefill 的总 token 数 |
| 总 Token 预算 | `max_batch_total_tokens` | 限制 batch 中活跃 token 总数 |

### 4.4 waiting_served_ratio 权衡

```
                   低延迟偏好 ←──────────→ 高吞吐偏好
                   (ratio=1.5)              (ratio=0.3)
                         │                      │
  行为:  立即组小batch    │        等待形成大batch  │
  延迟:  ★★★★★ (很低)    │    ★★☆ (较高)          │
  吞吐:  ★★☆ (较低)      │    ★★★★★ (很高)        │
  适用:  在线聊天         │    离线批量处理         │
```

`waiting_served_ratio` 是延迟与吞吐量的**平衡杠杆**：
- `ratio < 1.2`：等待中的 token 数少于当前处理中的 token 数 → 偏向吞吐
- `ratio >= 1.2`：等待中的请求足够多 → 偏向延迟（立即处理）

### 4.5 Prefill 与 Decode 的区别

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

## 5. 请求生命周期

### 5.1 完整时间线

```
Client                    Router (Rust)                  Model Server (Python)
  │                           │                                │
  │── POST /generate ────────▶│                                │
  │                           │── 1. 验证输入长度               │
  │                           │── 2. try_acquire() 过载保护    │
  │                           │── 3. 创建 QueueEntry            │
  │                           │── 4. queue.enqueue()            │
  │                           │── 5. notify_one()               │
  │◀── SSE stream ────────────│                                │
  │                           │                                │
  │                    [后台批处理任务]                          │
  │                           │── 6. wait_for_notify()          │
  │                           │── 7. collect_batch() 组 batch   │
  │                           │── 8. grpc Prefill ─────────────▶│
  │                           │                                │── Prefill 计算
  │                           │◀─────── PrefillResponse ───────│  (生成首token+KV Cache)
  │                           │                                │
  │                           │── 9. 发送首 token 到 response_tx│
  │◀── SSE: token_1 ─────────│                                │
  │                           │                                │
  │                    [Decode 循环]                            │
  │                           │── 10. grpc Decode ────────────▶│
  │                           │                                │── Decode 计算
  │                           │◀─────── DecodeResponse ────────│  (生成下一个token)
  │                           │── 11. 发送 token 到 response_tx │
  │◀── SSE: token_2 ─────────│                                │
  │                           │      ... (重复 step 10-11)      │
  │                           │                                │
  │                           │── 12. 检测 EOS / max_tokens     │
  │◀── SSE: [DONE] ──────────│                                │
  │                           │── 13. ClearCache ─────────────▶│── 释放 KV Cache
  │                           │                                │
```

### 5.2 请求状态机

```
                    ┌─────────┐
        HTTP 请求 ─▶│ WAITING │  等待被调度
                    └────┬────┘
                         │ collect_batch()
                         ▼
                    ┌─────────┐
                    │ PREFILL │  预填充 (处理 prompt)
                    └────┬────┘
                         │ prefill 完成
                         ▼
                    ┌─────────┐
              ┌────▶│ DECODE  │  逐个生成 token
              │     └────┬────┘
              │          │ decode 完成一步
              │          │ (未遇到 EOS 且未达 max_tokens)
              └──────────┘
                         │
                         │ EOS / max_tokens / error
                         ▼
                    ┌──────────┐
                    │ FINISHED │  完成，清理资源
                    └──────────┘
```

### 5.3 错误处理路径

```
验证失败 (输入为空/过长/参数非法)
    → 422 Unprocessable Entity

过载 (并发数超限)
    → 429 Too Many Requests (Semaphore::try_acquire 失败)

Prefill 阶段错误
    → 通过 response_tx 广播错误给 batch 中所有请求
    → 跳过 Decode，直接清理

Decode 阶段错误
    → 同上，batch 中剩余请求全部标记失败
```

---

## 6. Protobuf 协议设计

**文件**: `proto/generation.proto`

### 6.1 服务定义

```protobuf
service TextGenerationService {
  rpc Health(ModelInfoRequest) returns (ModelInfoResponse);
  rpc Prefill(BatchPrefillRequest) returns (BatchPrefillResponse);
  rpc Decode(BatchDecodeRequest) returns (BatchDecodeResponse);
  rpc ClearCache(ClearCacheRequest) returns (ClearCacheResponse);
}
```

### 6.2 批量接口设计

TGI 使用**批量接口**而非逐个请求发送，原因：

1. **减少 RPC 调用次数**：一次 gRPC 调用处理整个 batch
2. **GPU 批量计算**：Python 端可以将 batch 中所有请求拼接为一个 tensor
3. **原子性**：一个 batch 内的请求共享 forward pass

```
单请求接口 (不推荐):
  for each request:
    grpc_client.prefill_single(req)   ← N 次 RPC

批量接口 (TGI 采用):
  grpc_client.prefill_batch([req1, req2, ..., reqN])  ← 1 次 RPC
```

### 6.3 关键消息类型

```protobuf
// Prefill 请求 — 发送 prompt 和生成参数
message PrefillRequest {
  string request_id = 1;          // UUID 追踪
  repeated int32 input_ids = 2;   // tokenized prompt
  GenerationParameters params = 3; // 采样参数
  repeated int32 slot_ids = 4;    // KV cache slot
  int32 batch_id = 5;             // batch 内索引
}

// Prefill 响应 — 返回首 token + KV Cache 句柄
message PrefillResponse {
  string request_id = 1;
  repeated Token generated_tokens = 2;
  int64 cache_handle = 3;         // KV Cache 句柄 (后续 decode 使用)
  int32 prefill_duration_ms = 4;  // 性能指标
}

// Decode 请求 — 携带 KV Cache 句柄
message DecodeRequest {
  string request_id = 1;
  int32 token_id = 2;
  int64 cache_handle = 3;         // 从 Prefill 获取的句柄
  GenerationParameters params = 4;
}
```

### 6.4 代码生成

- **Rust 端**: `build.rs` 使用 `tonic-build` 在编译时从 proto 生成
- **Python 端**: `generate_proto.py` 使用 `grpc_tools.protoc` 生成

---

## 7. Rust Router 源码详解

### 7.1 项目结构

```
router/
├── Cargo.toml          # 依赖管理
├── build.rs            # 编译 proto → Rust 代码
└── src/
    ├── main.rs         # 入口：初始化、启动 Server + Scheduler
    ├── config.rs       # 配置管理 (环境变量)
    ├── server.rs       # HTTP Server (Axum)
    ├── queue.rs        # 请求队列 (mpsc + Notify)
    ├── scheduler.rs    # 批处理调度器 (核心)
    └── infer.rs        # gRPC 客户端封装
```

### 7.2 main.rs — 启动流程

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. 初始化日志
    tracing_subscriber::init();

    // 2. 加载配置
    let config = RouterConfig::from_env()?;

    // 3. 创建 gRPC 客户端
    let grpc_client = GrpcClient::connect(&grpc_addr).await?;

    // 4. 创建共享组件
    let queue = Arc::new(RequestQueue::new());
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
    let scheduler = Arc::new(BatchingScheduler::new(config, grpc_client, queue));

    // 5. spawn 后台批处理任务 (永不结束的 tokio task)
    tokio::spawn(async move { scheduler.run().await });

    // 6. 启动 HTTP Server
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
}
```

**关键设计**：
- `Arc<RequestQueue>` — 多个 HTTP 线程和后台任务共享同一个队列
- `Arc<Semaphore>` — 所有 HTTP handler 共享同一个并发限制器
- `tokio::spawn` — 后台任务与 HTTP Server 并行运行

### 7.3 server.rs — HTTP 处理

**验证逻辑** (`validate_request`):

```
检查项:
  ├─ 输入非空
  ├─ token 数 ≤ max_input_length (默认 4096)
  ├─ temperature ∈ [0.0, 2.0]
  ├─ 总 token 数 ≤ max_total_tokens (默认 8192)
  └─ max_new_tokens ≤ 2048
```

**SSE 流构建** (`build_sse_stream`):

```rust
fn build_sse_stream(rx, permit, queue_time) -> impl Stream<Item = Event> {
    UnboundedReceiverStream::new(rx)
        .map(move |token| {
            let response = GenerateStreamResponse {
                token: TokenInfo { id, text, special },
                generated_text: Some(accumulated_text),
                details: if finished { Some(details) } else { None },
            };
            Event::default().data(serde_json::to_string(&response))
        })
    // permit 在此 drop → 释放信号量
}
```

**permit 的生命周期管理**：
- permit 被移入 stream 的闭包中
- stream 结束时（客户端断开或生成完成），permit 自动 drop
- drop 时自动释放 Semaphore，允许新请求进入

### 7.4 queue.rs — 请求队列

**原子计数器的使用**：

```rust
pub struct RequestQueue {
    tx: mpsc::UnboundedSender<QueueEntry>,
    rx: parking_lot::Mutex<Option<mpsc::UnboundedReceiver<QueueEntry>>>,
    notify: Notify,
    waiting_tokens: AtomicUsize,  // 原子操作，无锁
}
```

**为什么用 `parking_lot::Mutex<Option<...>>`？**
- `mpsc::UnboundedReceiver` 不能被 clone
- 只有后台任务需要 receiver
- `take()` 确保 receiver 只被取走一次

**Notify 机制**：

```
HTTP Thread                    Batching Task
    │                              │
    │── enqueue(entry) ──▶         │ (sleeping on notify.notified())
    │── notify.notify_one() ──────▶│ (wakes up)
    │                              │── try_recv() 收集所有待处理请求
```

### 7.5 scheduler.rs — 调度器核心（详解）

**BatchingScheduler 结构**：

```rust
pub struct BatchingScheduler {
    config: RouterConfig,           // 调度参数
    grpc_client: GrpcClient,        // gRPC 连接
    queue: Arc<RequestQueue>,       // 请求队列 (共享)
}
```

**RequestState 结构**：

```rust
struct RequestState {
    request_id: String,             // UUID
    input_ids: Vec<i32>,            // tokenized prompt
    max_new_tokens: u32,            // 最大生成数
    generated_count: u32,           // 已生成计数
    cache_handle: i64,              // KV Cache 句柄
    response_tx: UnboundedSender,   // 发回 HTTP 线程的通道
    queue_time: DateTime<Utc>,      // 入队时间
    is_finished: bool,              // 完成标志
}
```

**Prefill 流程** (`run_prefill`):

```
1. 遍历 batch 中所有请求
2. 构造 BatchPrefillRequest (含所有请求的 input_ids + params)
3. 一次 gRPC 调用发送到 Python Server
4. Python Server 返回:
   - cache_handle: KV Cache 句柄
   - generated_tokens: 首个生成的 token
   - prefill_duration_ms: 耗时
5. 更新 RequestState (cache_handle, generated_count)
6. 通过 response_tx 发送首 token 到 HTTP 线程
```

**Decode 循环** (`run_decode`):

```
1. 过滤出 is_finished == false 的请求
2. 构造 BatchDecodeRequest (含所有活跃请求的 cache_handle)
3. 一次 gRPC 调用
4. 每个请求获得一个 token
5. 检查停止条件:
   - token.id == EOS → finish_reason = "eos_token"
   - generated_count >= max_new_tokens → finish_reason = "length"
6. 通过 response_tx 发送 token
7. 如果所有请求都 finished → 退出循环
```

**失败处理** (`fail_all_requests`):

```rust
fn fail_all_requests(&self, requests: &[RequestState], error: &str) {
    for req in requests {
        let _ = req.response_tx.send(QueueToken {
            is_finished: true,
            finish_reason: Some(format!("error: {}", error)),
            ..Default::default()
        });
    }
}
```

所有失败请求都通过 `response_tx` 通知 HTTP 线程，确保客户端不会无限等待。

### 7.6 infer.rs — gRPC 客户端

```rust
pub struct GrpcClient {
    inner: Arc<parking_lot::Mutex<TextGenerationServiceClient<Channel>>>,
}
```

**为什么用 `Arc<Mutex<...>>` 包装？**
- tonic 的 client 需要 `&mut self` 来调用方法
- 但多个地方可能需要持有 client 引用
- `parking_lot::Mutex` 比标准库 `Mutex` 更快（无中毒检测）

---

## 8. Python Model Server 源码详解

### 8.1 项目结构

```
model_server/
├── pyproject.toml        # Python 项目配置
├── generate_proto.py     # 编译 proto → Python 代码
├── grpc_server.py        # gRPC 服务实现 (入口)
├── model_engine.py       # 模型推理引擎 (核心)
└── test_client.py        # gRPC 测试客户端
```

### 8.2 model_engine.py — 推理引擎

**ModelEngine 结构**：

```python
class ModelEngine:
    def __init__(self, model_id, max_sequence_length, max_batch_size, vocab_size, eos_token_id):
        self._cache: OrderedDict[int, CacheEntry] = OrderedDict()
        self._handle_counter = 1
```

**KV Cache 管理**：

```python
@dataclass
class CacheEntry:
    handle: int                    # 唯一句柄
    request_id: str                # 请求 ID
    key_cache: List[torch.Tensor]  # K 矩阵 (每层一个)
    value_cache: List[torch.Tensor]# V 矩阵 (每层一个)
    generated_ids: List[int]       # 已生成的 token 列表
    is_finished: bool              # 是否完成
```

**Prefill 实现**：

```python
def prefill(self, request_id, input_ids, max_new_tokens, temperature, ...):
    # 1. 分配 KV Cache 句柄
    handle = self._handle_counter
    self._handle_counter += 1
    
    # 2. 模拟 Prefill 计算延迟
    time.sleep(0.001 * len(input_ids))  # 每 token 1ms
    
    # 3. 生成第一个 token (模拟)
    first_token = self._generate_next_token(input_ids, temperature, do_sample)
    
    # 4. 创建 CacheEntry 并保存
    entry = CacheEntry(handle=handle, generated_ids=list(input_ids) + [first_token])
    entry.key_cache = [torch.randn(1, 64)]  # 模拟 KV Cache
    self._cache[handle] = entry
    
    return handle, [{"id": first_token, "text": ...}], duration_ms
```

**Decode 实现**：

```python
def decode(self, request_id, cache_handle, ...):
    # 1. 从缓存中获取 KV Cache
    entry = self._cache[cache_handle]
    
    # 2. 模拟 Decode 计算延迟
    time.sleep(0.002)  # 2ms
    
    # 3. 生成下一个 token
    next_token = self._generate_next_token(entry.generated_ids, ...)
    entry.generated_ids.append(next_token)
    
    # 4. 检查 EOS
    if next_token == self.eos_token_id:
        finish_reason = "eos_token"
        entry.is_finished = True
    
    return {"id": next_token, "text": ...}, finish_reason, duration_ms
```

**模拟 token 生成**：

```python
def _generate_next_token(self, context_ids, temperature, do_sample):
    # 基于最近 4 个 token 的 hash 生成伪随机 token
    recent = context_ids[-4:]
    hash_val = sum(t * (i+1) * 2654435761 for i, t in enumerate(recent))
    token = (hash_val % (self.vocab_size - 1)) + 1
    return token
```

### 8.3 grpc_server.py — gRPC 服务

**TextGenerationServicer** 继承自 proto 生成的基类：

```python
class TextGenerationServicer(rpc.TextGenerationServiceServicer):
    def __init__(self, engine: ModelEngine):
        self.engine = engine

    def Prefill(self, request, context):
        # 遍历 batch 中所有请求
        for req in request.requests:
            handle, tokens, duration = self.engine.prefill(...)
            response = pb.PrefillResponse(
                request_id=req.request_id,
                cache_handle=handle,
                prefill_duration_ms=duration,
            )
            # 填充生成的 token
            for t in tokens:
                response.generated_tokens.append(pb.Token(id=t["id"], text=t["text"]))
            responses.append(response)
        
        return pb.BatchPrefillResponse(batch_id=request.batch_id, responses=responses)
```

**线程池配置**：

```python
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),  # 10 个工作线程
    options=[
        ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ],
)
```

---

## 9. 配置与调优

### 9.1 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ROUTER_HOST` | `0.0.0.0` | HTTP 监听地址 |
| `ROUTER_PORT` | `3000` | HTTP 端口 |
| `MODEL_SERVER_HOST` | `127.0.0.1` | Python 服务地址 |
| `MODEL_SERVER_PORT` | `50051` | gRPC 端口 |
| `MAX_CONCURRENT_REQUESTS` | `128` | 最大并发请求数 |
| `MAX_BATCH_SIZE` | `32` | 单 batch 最大请求数 |
| `MAX_BATCH_PREFILL_TOKENS` | `4096` | 单 batch prefill token 上限 |
| `MAX_BATCH_TOTAL_TOKENS` | `16384` | 单 batch 总 token 上限 |
| `MAX_WAITING_TOKENS` | `20` | 触发组 batch 的等待 token 阈值 |
| `WAITING_SERVED_RATIO` | `1.2` | 延迟/吞吐平衡参数 |
| `MAX_INPUT_LENGTH` | `4096` | 最大输入长度 |
| `MAX_TOTAL_TOKENS` | `8192` | 单请求最大总 token 数 |

### 9.2 调优建议

| 场景 | 推荐配置 |
|------|---------|
| **在线聊天 (低延迟)** | `WAITING_SERVED_RATIO=1.5`, `MAX_BATCH_SIZE=4`, `MAX_CONCURRENT_REQUESTS=256` |
| **批量处理 (高吞吐)** | `WAITING_SERVED_RATIO=0.3`, `MAX_BATCH_SIZE=32`, `MAX_BATCH_PREFILL_TOKENS=8192` |
| **GPU 内存有限** | 减小 `MAX_BATCH_TOTAL_TOKENS`，增大 `MAX_CONCURRENT_REQUESTS` |

### 9.3 延迟分解

```
总延迟 = 排队时间 + Prefill 时间 + (Decode 时间 × 生成 token 数)

排队时间:  由并发数和 waiting_served_ratio 决定
Prefill:   与 prompt 长度正相关 (O(n²) attention)
Decode:    每次 ~2-5ms，与生成 token 数线性相关
```

---

## 10. 运行指南

### 10.1 环境准备

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

### 10.2 编译 proto

```bash
# 生成 Python gRPC 代码
cd light_tgi/model_server
python generate_proto.py
```

### 10.3 启动服务

**终端 1 — 启动 Python Model Server**：
```bash
cd light_tgi/model_server
python grpc_server.py --port 50051
```

**终端 2 — 编译并启动 Rust Router**：
```bash
cd light_tgi/router
cargo run --release
```

### 10.4 测试

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

## 11. 性能分析

### 11.1 关键指标

| 指标 | 说明 | 获取方式 |
|------|------|---------|
| 排队时间 | 请求在队列中的等待时间 | `x-queue-time` 响应头 |
| Prefill 时间 | 处理 prompt 的耗时 | `prefill_duration_ms` 字段 |
| Decode 时间 | 每步生成 token 的耗时 | `decode_duration_ms` 字段 |
| 吞吐量 | 每秒生成的 token 数 | 总 tokens / 总时间 |
| 并发数 | 当前活跃请求数 | Semaphore available_permits |

### 11.2 扩展点

1. **PagedAttention**：将 KV Cache 分页管理，减少内存碎片
2. **Prefix Caching**：缓存共享的 prompt 前缀，避免重复 prefill
3. **Speculative Decoding**：用小模型"猜测"多个 token，大模型验证
4. **Tensor Parallel**：多 GPU 分片推理
5. **量化支持**：INT8/INT4 权重压缩

### 11.3 从教学版到生产版

```
Light TGI (教学版)              →    TGI (生产版)
─────────────────────────────────────────────────
模拟 token 生成                 →    真实模型推理 (FlashAttention)
简化 KV Cache                  →    PagedAttention / FlashInfer
单线程 Python Server           →    多 GPU Tensor Parallel
固定参数                       →    CLI 参数 + 配置文件
无监控                         →    Prometheus metrics
无鉴权                         →    API Key / OAuth
```

---

## 附录

### A. 文件清单

```
light_tgi/
├── proto/
│   └── generation.proto          # Protobuf 协议定义
├── router/
│   ├── Cargo.toml                # Rust 依赖
│   ├── build.rs                  # Proto 编译脚本
│   └── src/
│       ├── main.rs               # 入口
│       ├── config.rs             # 配置
│       ├── server.rs             # HTTP Server
│       ├── queue.rs              # 请求队列
│       ├── scheduler.rs          # 调度器 (核心)
│       └── infer.rs              # gRPC 客户端
├── model_server/
│   ├── pyproject.toml            # Python 依赖
│   ├── generate_proto.py         # Proto 编译
│   ├── grpc_server.py            # gRPC 服务
│   ├── model_engine.py           # 推理引擎
│   └── test_client.py            # 测试客户端
├── test_api.py                   # HTTP API 测试
└── DESIGN_DOC.md                 # 本文档
```

### B. 参考资料

- [TGI GitHub](https://github.com/huggingface/text-generation-inference)
- [TGI Architecture](https://hugging-face.cn/docs/text-generation-inference/architecture)
- [Continuous Batching 论文](https://arxiv.org/abs/2308.09596) (vLLM)
- [Axum 框架](https://github.com/tokio-rs/axum)
- [Tonic gRPC](https://github.com/hyperium/tonic)
