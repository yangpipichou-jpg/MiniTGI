"""
gRPC 模型服务器 (生产级 v4 — Continuous Batching)

架构升级:
  v3: StreamGenerate RPC — 单请求流式, Session 独立调用
  v4: ★ BatchStreamGenerate RPC — 双向流, Python 端做真正的 batch forward
       ★ PrefixCacheLookup / PrefixCacheStore — Prefix Sharing

Python 环境: F:\ProgramData\anaconda3\python.exe
"""

import time
import logging
import argparse
import sys
import os
import threading
import queue
from concurrent import futures

import grpc

sys.path.insert(0, os.path.dirname(__file__))

try:
    import generation_pb2 as pb
    import generation_pb2_grpc as rpc
except ImportError:
    print("错误: 未找到生成的 proto 代码。请先运行 generate_proto.py")
    print("  F:\\ProgramData\\anaconda3\\python.exe generate_proto.py")
    sys.exit(1)

from model_engine import ModelEngine
from batch_scheduler import BatchScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpc_server")


class TextGenerationServicer(rpc.TextGenerationServiceServicer):
    """gRPC 服务实现 (v4: Continuous Batching + Prefix Sharing)"""

    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.start_time = time.time()

        # ★ BatchScheduler: 生产级 Continuous Batching
        self.scheduler = BatchScheduler(
            engine=engine,
            max_batch_size=engine.max_batch_size,
            max_batch_prefill_tokens=int(os.environ.get("MAX_BATCH_PREFILL_TOKENS", "4096")),
            max_active_requests=int(os.environ.get("MAX_ACTIVE_REQUESTS", "32")),
            prefix_cache_size=int(os.environ.get("PREFIX_CACHE_SIZE", "64")),
        )

    def _cleanup_stale_state(self):
        """
        清理上一次 BatchStreamGenerate 连接残留的状态。

        当 Rust Router 重连时，旧的 active/pending 请求和 engine cache
        可能残留，导致跨连接污染（输出混乱）。
        """
        # 清理 scheduler 中的活跃和等待请求
        with self.scheduler.active_lock:
            for rid, req in list(self.scheduler.active.items()):
                try:
                    self.engine.clear_cache([req.cache_handle])
                except Exception:
                    pass
            self.scheduler.active.clear()

        with self.scheduler.pending_lock:
            self.scheduler.pending.clear()

        # 清理 engine 中残留的 cache (防御性清理)
        stale_handles = list(self.engine._cache.keys())
        if stale_handles:
            logger.warning(
                f"[BatchStreamGenerate] 清理 {len(stale_handles)} 个残留 cache: "
                f"{stale_handles}"
            )
            self.engine.clear_cache(stale_handles)

        logger.info(
            f"[BatchStreamGenerate] 状态清理完成: "
            f"active={self.scheduler.active_count}, "
            f"pending={self.scheduler.pending_count}, "
            f"cached={len(self.engine._cache)}"
        )

    # ================================================================
    # Health
    # ================================================================

    def Health(self, request: pb.ModelInfoRequest, context) -> pb.ModelInfoResponse:
        return pb.ModelInfoResponse(
            model_id=self.engine.model_id,
            max_sequence_length=self.engine.max_sequence_length,
            max_batch_size=self.engine.max_batch_size,
            vocab_size=self.engine.vocab_size,
            eos_token_id=self.engine.eos_token_id,
            device=self.engine.device,
        )

    # ================================================================
    # Tokenize
    # ================================================================

    def Tokenize(self, request: pb.TokenizeRequest, context) -> pb.TokenizeResponse:
        token_ids, count = self.engine.tokenize(request.text)
        return pb.TokenizeResponse(token_ids=token_ids, token_count=count)

    # ================================================================
    # ★ BatchStreamGenerate — 双向流 Continuous Batching (v4 核心)
    # ================================================================

    def BatchStreamGenerate(self, request_iterator, context):
        """
        双向流 Continuous Batching

        Rust Router 持续发送 BatchCommand (ADD_REQUEST / CANCEL_REQUEST),
        Python 端批量调度:
          - 新请求 → 加入 pending 队列
          - 定时 step() → batch prefill + batch decode
          - 流式返回所有请求的 token

        这是真正的 Continuous Batching:
          - 多个请求共享同一次 GPU forward
          - 新请求可随时加入
          - GPU 利用率始终最高
        """
        logger.info("[BatchStreamGenerate] 双向流连接建立")

        # ★ 清理上一次连接残留的状态 (防止跨连接污染)
        self._cleanup_stale_state()

        stop_event = threading.Event()
        response_queue = queue.Queue()
        request_ingest_queue = queue.Queue()

        # 启动调度循环线程
        def scheduler_loop():
            """独立线程运行调度器"""
            try:
                while not stop_event.is_set():
                    # 1. 从 Rust 侧接收新请求
                    try:
                        while True:
                            cmd = request_ingest_queue.get_nowait()
                            for req in cmd.get("new_requests", []):
                                params = req.get("params", {})
                                self.scheduler.add_request(
                                    request_id=req["request_id"],
                                    input_text=req["input_text"],
                                    max_new_tokens=params.get("max_new_tokens", 100),
                                    temperature=params.get("temperature", 1.0),
                                    top_p=params.get("top_p", 1.0),
                                    top_k=params.get("top_k", 0),
                                    do_sample=params.get("do_sample", False),
                                )
                            for rid in cmd.get("cancel_ids", []):
                                self.scheduler.cancel_request(rid)
                    except queue.Empty:
                        pass

                    # 2. 执行一步调度
                    if self.scheduler.pending_count > 0 or self.scheduler.active_count > 0:
                        results = self.scheduler.step()
                        if results:
                            response_queue.put(results)
                    else:
                        time.sleep(0.005)  # 5ms idle
            except Exception as e:
                logger.error(f"[BatchStreamGenerate] 调度线程异常: {e}")
                response_queue.put(None)  # 信号: 异常退出

        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()

        # 启动 Rust → Python 接收线程
        def ingest_loop():
            """从 Rust 侧接收 BatchCommand"""
            try:
                for cmd in request_iterator:
                    new_requests = []
                    cancel_ids = []

                    if cmd.command_type == pb.BatchCommandType.ADD_REQUEST:
                        for req in cmd.new_requests:
                            params = req.params
                            new_requests.append({
                                "request_id": req.request_id,
                                "input_text": req.input_text,
                                "params": {
                                    "max_new_tokens": params.max_new_tokens if params else 100,
                                    "temperature": params.temperature if params else 1.0,
                                    "top_p": params.top_p if params else 1.0,
                                    "top_k": params.top_k if params else 0,
                                    "do_sample": params.do_sample if params else False,
                                },
                            })
                    elif cmd.command_type == pb.BatchCommandType.CANCEL_REQUEST:
                        cancel_ids = list(cmd.cancel_request_ids)

                    if new_requests or cancel_ids:
                        request_ingest_queue.put({
                            "new_requests": new_requests,
                            "cancel_ids": cancel_ids,
                        })

                # Rust 侧 stream 结束
                logger.info("[BatchStreamGenerate] Rust 侧 stream 关闭")
            except Exception as e:
                logger.error(f"[BatchStreamGenerate] 接收线程异常: {e}")
            finally:
                stop_event.set()

        ingest_thread = threading.Thread(target=ingest_loop, daemon=True)
        ingest_thread.start()

        # 主线程: 从 response_queue 取出结果, yield 给 Rust
        try:
            while not stop_event.is_set() or not response_queue.empty():
                try:
                    results = response_queue.get(timeout=0.05)
                    if results is None:
                        break  # 调度线程异常

                    # 构建 BatchStreamResponse
                    responses = []
                    for request_id, token_dict, finish_reason in results:
                        grpc_reason = pb.FinishReason.NONE
                        if finish_reason == "eos_token":
                            grpc_reason = pb.FinishReason.EOS_TOKEN
                        elif finish_reason == "length":
                            grpc_reason = pb.FinishReason.MAX_TOKENS
                        elif finish_reason == "error":
                            grpc_reason = pb.FinishReason.ERROR_REASON

                        resp = pb.StreamGenerateResponse(
                            request_id=request_id,
                            finish_reason=grpc_reason,
                            step=0,
                            duration_ms=0,
                            cache_handle=0,
                        )
                        if token_dict:
                            resp.generated_token.CopyFrom(pb.Token(
                                id=token_dict["id"],
                                text=token_dict["text"],
                                logprob=token_dict.get("logprob", 0.0),
                                special=token_dict.get("special", False),
                            ))
                        responses.append(resp)

                    yield pb.BatchStreamResponse(
                        batch_id=0,
                        tokens=responses,
                        active_requests=self.scheduler.active_count,
                        batch_decode_ms=0,
                    )

                except queue.Empty:
                    continue

        except Exception as e:
            logger.error(f"[BatchStreamGenerate] 主循环异常: {e}")
        finally:
            stop_event.set()
            scheduler_thread.join(timeout=5)
            ingest_thread.join(timeout=5)
            logger.info(
                f"[BatchStreamGenerate] 连接关闭, "
                f"stats={self.scheduler.get_stats()}"
            )

    # ================================================================
    # ★ Prefix Sharing RPCs (v4 新增)
    # ================================================================

    def PrefixCacheLookup(self, request: pb.PrefixCacheLookupRequest, context) -> pb.PrefixCacheLookupResponse:
        """查找前缀缓存"""
        entry = self.scheduler.prefix_cache.lookup(
            request.prefix_text, request.min_match_tokens
        )
        if entry:
            return pb.PrefixCacheLookupResponse(
                found=True,
                shared_cache_handle=entry["cache_handle"],
                matched_tokens=entry["token_count"],
            )
        return pb.PrefixCacheLookupResponse(found=False)

    def PrefixCacheStore(self, request: pb.PrefixCacheStoreRequest, context) -> pb.PrefixCacheStoreResponse:
        """存储前缀缓存"""
        self.scheduler.prefix_cache.store(
            request.prefix_text, request.cache_handle,
            request.token_count, [],
        )
        return pb.PrefixCacheStoreResponse(success=True)

    # ================================================================
    # ★ StreamGenerate — 单请求流式 (保留兼容)
    # ================================================================

    def StreamGenerate(self, request: pb.StreamGenerateRequest, context):
        """单请求流式生成 (保留兼容, 用于调试)"""
        request_id = request.request_id
        input_text = request.input_text
        params = request.params

        logger.info(f"[StreamGenerate] 开始: request={request_id}")

        try:
            step = 0
            start = time.time()
            cache_handle, tokens, _ = self.engine.prefill(
                request_id=request_id,
                input_text=input_text,
                max_new_tokens=params.max_new_tokens if params else 100,
                temperature=params.temperature if params else 1.0,
                top_p=params.top_p if params else 1.0,
                top_k=params.top_k if params else 0,
                do_sample=params.do_sample if params else False,
            )
            duration_ms = int((time.time() - start) * 1000)

            for token in tokens:
                is_eos = token["id"] == self.engine.eos_token_id
                yield pb.StreamGenerateResponse(
                    request_id=request_id,
                    generated_token=pb.Token(
                        id=token["id"], text=token["text"],
                        logprob=token.get("logprob", 0.0),
                        special=token.get("special", False),
                    ),
                    finish_reason=pb.FinishReason.EOS_TOKEN if is_eos else pb.FinishReason.NONE,
                    step=step, duration_ms=duration_ms, cache_handle=cache_handle,
                )
                if is_eos:
                    self.engine.clear_cache([cache_handle])
                    return

            step += 1
            while True:
                start = time.time()
                token_dict, finish_reason, _ = self.engine.decode(
                    request_id=request_id, cache_handle=cache_handle,
                    max_new_tokens=params.max_new_tokens if params else 100,
                    temperature=params.temperature if params else 1.0,
                    top_p=params.top_p if params else 1.0,
                    top_k=params.top_k if params else 0,
                    do_sample=params.do_sample if params else False,
                )
                duration_ms = int((time.time() - start) * 1000)

                if token_dict is None:
                    break

                reason_map = {
                    "eos_token": pb.FinishReason.EOS_TOKEN,
                    "length": pb.FinishReason.MAX_TOKENS,
                    "stop_sequence": pb.FinishReason.STOP_SEQUENCE,
                    "error": pb.FinishReason.ERROR_REASON,
                }
                grpc_reason = reason_map.get(finish_reason, pb.FinishReason.NONE) if finish_reason else pb.FinishReason.NONE

                yield pb.StreamGenerateResponse(
                    request_id=request_id,
                    generated_token=pb.Token(
                        id=token_dict["id"], text=token_dict["text"],
                        logprob=token_dict.get("logprob", 0.0),
                        special=token_dict.get("special", False),
                    ),
                    finish_reason=grpc_reason, step=step,
                    duration_ms=duration_ms, cache_handle=cache_handle,
                )

                if finish_reason:
                    self.engine.clear_cache([cache_handle])
                    return
                step += 1

        except Exception as e:
            logger.error(f"[StreamGenerate] 错误: request={request_id}, error={e}")
            yield pb.StreamGenerateResponse(
                request_id=request_id,
                finish_reason=pb.FinishReason.ERROR_REASON,
                step=-1, duration_ms=0, cache_handle=0,
            )

    # ================================================================
    # Prefill (批量, 保留兼容)
    # ================================================================

    def Prefill(self, request: pb.BatchPrefillRequest, context) -> pb.BatchPrefillResponse:
        logger.info(f"收到 Prefill: batch_id={request.batch_id}, requests={len(request.requests)}")

        batch_requests = []
        for req in request.requests:
            params = req.params
            batch_requests.append({
                "request_id": req.request_id,
                "input_text": req.input_text,
                "params": {
                    "max_new_tokens": params.max_new_tokens if params else 100,
                    "temperature": params.temperature if params else 1.0,
                    "top_p": params.top_p if params else 1.0,
                    "top_k": params.top_k if params else 0,
                    "do_sample": params.do_sample if params else False,
                },
            })

        results = self.engine.batch_prefill(batch_requests)

        responses = []
        for i, (cache_handle, generated_tokens, duration_ms) in enumerate(results):
            response = pb.PrefillResponse(
                request_id=batch_requests[i]["request_id"],
                cache_handle=cache_handle,
                prefill_duration_ms=duration_ms,
                prompt_token_count=self.engine.tokenize(batch_requests[i]["input_text"])[1],
            )
            for token in generated_tokens:
                response.generated_tokens.append(pb.Token(
                    id=token["id"], text=token["text"],
                    logprob=token.get("logprob", 0.0), special=token.get("special", False),
                ))
            responses.append(response)

        return pb.BatchPrefillResponse(batch_id=request.batch_id, responses=responses)

    # ================================================================
    # Decode (批量, 保留兼容)
    # ================================================================

    def Decode(self, request: pb.BatchDecodeRequest, context) -> pb.BatchDecodeResponse:
        responses = []
        for req in request.requests:
            params = req.params
            token_dict, finish_reason, duration_ms = self.engine.decode(
                request_id=req.request_id,
                cache_handle=req.cache_handle,
                max_new_tokens=params.max_new_tokens if params else 100,
                temperature=params.temperature if params else 1.0,
                top_p=params.top_p if params else 1.0,
                top_k=params.top_k if params else 0,
                do_sample=params.do_sample if params else False,
            )

            response = pb.DecodeResponse(request_id=req.request_id, decode_duration_ms=duration_ms)
            reason_map = {
                "eos_token": pb.FinishReason.EOS_TOKEN, "length": pb.FinishReason.MAX_TOKENS,
                "stop_sequence": pb.FinishReason.STOP_SEQUENCE, "error": pb.FinishReason.ERROR_REASON,
            }
            response.finish_reason = reason_map.get(finish_reason, pb.FinishReason.NONE) if finish_reason else pb.FinishReason.NONE

            if token_dict:
                response.generated_token.CopyFrom(pb.Token(
                    id=token_dict["id"], text=token_dict["text"],
                    logprob=token_dict.get("logprob", 0.0), special=token_dict.get("special", False),
                ))
            responses.append(response)

        return pb.BatchDecodeResponse(batch_id=request.batch_id, responses=responses)

    # ================================================================
    # ClearCache
    # ================================================================

    def ClearCache(self, request: pb.ClearCacheRequest, context) -> pb.ClearCacheResponse:
        success = self.engine.clear_cache(list(request.cache_handles))
        return pb.ClearCacheResponse(success=success)


def serve(
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cpu",
    dtype: str = "auto",
):
    logger.info("=" * 60)
    logger.info("Light TGI Model Server v3 (事件驱动架构)")
    logger.info("=" * 60)

    engine = ModelEngine(
        model_id=model_id,
        max_sequence_length=int(os.environ.get("MAX_SEQUENCE_LENGTH", "4096")),
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "8")),
        device=device,
        dtype=dtype,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 10000),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.max_pings_without_data", 0),
        ],
    )

    servicer = TextGenerationServicer(engine)
    rpc.add_TextGenerationServiceServicer_to_server(servicer, server)

    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    logger.info(f"监听地址: {addr}")
    logger.info(f"模型: {engine.model_id}")
    logger.info(f"设备: {engine.device}")
    logger.info(f"★ StreamGenerate RPC 已启用 (异步流式)")
    logger.info("按 Ctrl+C 停止服务")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("收到关闭信号，正在优雅退出...")
        server.stop(grace=5)
        logger.info("服务已停止")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Light TGI Model Server v3")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    args = parser.parse_args()

    serve(host=args.host, port=args.port, max_workers=args.workers,
          model_id=args.model_id, device=args.device, dtype=args.dtype)
