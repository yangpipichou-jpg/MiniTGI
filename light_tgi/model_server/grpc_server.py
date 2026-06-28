"""
gRPC 模型服务器 (事件驱动版本 v3)

架构升级变更:
  旧: Prefill + Decode 两个独立 RPC, Rust 端控制循环
  新: ★ StreamGenerate RPC — Session 发一次请求, Python 流式返回所有 token
       Python 端内部控制 Prefill→Decode 循环, Rust 端只需消费 stream

Python 环境: F:\ProgramData\anaconda3\python.exe
"""

import time
import logging
import argparse
import sys
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpc_server")


class TextGenerationServicer(rpc.TextGenerationServiceServicer):
    """gRPC 服务实现 (v3: 新增 StreamGenerate)"""

    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.start_time = time.time()

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
    # ★ StreamGenerate — 异步流式生成 (事件驱动架构核心)
    # ================================================================

    def StreamGenerate(self, request: pb.StreamGenerateRequest, context):
        """
        流式生成: 接收一次请求, 持续 yield token 直到完成

        Session 无需再手动控制 Prefill→Decode 循环,
        Python 端内部完成整个生成流程并通过 stream 返回。

        流程:
          1. Prefill → yield 首 token + cache_handle
          2. Decode 循环 → yield 每个 token
          3. 遇到 EOS 或 max_new_tokens → yield 最后一个 token (finish_reason 非空)
          4. 清理 KV Cache
        """
        request_id = request.request_id
        input_text = request.input_text
        params = request.params

        logger.info(f"[StreamGenerate] 开始: request={request_id}")

        try:
            # Step 1: Prefill
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

            # Yield 首 token
            for token in tokens:
                is_eos = token["id"] == self.engine.eos_token_id
                yield pb.StreamGenerateResponse(
                    request_id=request_id,
                    generated_token=pb.Token(
                        id=token["id"],
                        text=token["text"],
                        logprob=token.get("logprob", 0.0),
                        special=token.get("special", False),
                    ),
                    finish_reason=pb.FinishReason.EOS_TOKEN if is_eos else pb.FinishReason.NONE,
                    step=step,
                    duration_ms=duration_ms,
                    cache_handle=cache_handle,
                )

                if is_eos:
                    logger.info(f"[StreamGenerate] EOS 在首 token: request={request_id}")
                    self.engine.clear_cache([cache_handle])
                    return

            step += 1

            # Step 2: Decode 循环
            while True:
                start = time.time()
                token_dict, finish_reason, _ = self.engine.decode(
                    request_id=request_id,
                    cache_handle=cache_handle,
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
                        id=token_dict["id"],
                        text=token_dict["text"],
                        logprob=token_dict.get("logprob", 0.0),
                        special=token_dict.get("special", False),
                    ),
                    finish_reason=grpc_reason,
                    step=step,
                    duration_ms=duration_ms,
                    cache_handle=cache_handle,
                )

                if finish_reason:
                    logger.info(
                        f"[StreamGenerate] 完成: request={request_id}, "
                        f"reason={finish_reason}, steps={step}"
                    )
                    self.engine.clear_cache([cache_handle])
                    return

                step += 1

        except Exception as e:
            logger.error(f"[StreamGenerate] 错误: request={request_id}, error={e}")
            yield pb.StreamGenerateResponse(
                request_id=request_id,
                finish_reason=pb.FinishReason.ERROR_REASON,
                step=-1,
                duration_ms=0,
                cache_handle=0,
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
