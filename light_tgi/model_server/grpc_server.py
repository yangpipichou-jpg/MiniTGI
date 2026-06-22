"""
gRPC 模型服务器

职责：
1. 启动 gRPC 服务监听
2. 实现 TextGenerationService 的所有 RPC 方法
3. 将请求转发给 ModelEngine 处理
4. 返回格式化的响应

这是 Python 端的主入口，对应 TGI 的 Model Server。
"""

import time
import logging
import argparse
import sys
import os
from concurrent import futures
from typing import Dict

import grpc

# 添加 proto 生成的代码路径
sys.path.insert(0, os.path.dirname(__file__))

# 尝试导入生成的 proto 代码
try:
    import generation_pb2 as pb
    import generation_pb2_grpc as rpc
except ImportError:
    print("错误: 未找到生成的 proto 代码。请先运行 generate_proto.py")
    print("  python generate_proto.py")
    sys.exit(1)

from model_engine import ModelEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpc_server")


class TextGenerationServicer(rpc.TextGenerationServiceServicer):
    """
    gRPC 服务实现

    实现 generation.proto 中定义的 TextGenerationService。
    所有请求委托给 ModelEngine 处理。
    """

    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.start_time = time.time()

    # ================================================================
    # Health - 健康检查 / 模型信息
    # ================================================================

    def Health(self, request: pb.ModelInfoRequest, context) -> pb.ModelInfoResponse:
        """返回模型信息"""
        logger.debug("收到 Health 请求")
        return pb.ModelInfoResponse(
            model_id=self.engine.model_id,
            max_sequence_length=self.engine.max_sequence_length,
            max_batch_size=self.engine.max_batch_size,
            vocab_size=self.engine.vocab_size,
            eos_token_id=self.engine.eos_token_id,
        )

    # ================================================================
    # Prefill - 批量预填充
    # ================================================================

    def Prefill(
        self, request: pb.BatchPrefillRequest, context
    ) -> pb.BatchPrefillResponse:
        """
        批量预填充

        对 batch 中的所有请求执行 prefill：
        1. 处理每个请求的 input_ids
        2. 生成首个 token
        3. 分配 KV Cache
        4. 返回结果
        """
        logger.info(
            f"收到 Prefill 请求: batch_id={request.batch_id}, "
            f"requests={len(request.requests)}"
        )

        responses = []
        total_tokens = 0

        for req in request.requests:
            # 提取参数
            params = req.params
            max_new_tokens = params.max_new_tokens if params else 100
            temperature = params.temperature if params else 1.0
            top_p = params.top_p if params else 1.0
            top_k = params.top_k if params else 0
            do_sample = params.do_sample if params else False

            total_tokens += len(req.input_ids)

            # 调用引擎
            cache_handle, generated_tokens, duration_ms = self.engine.prefill(
                request_id=req.request_id,
                input_ids=list(req.input_ids),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )

            # 构建响应
            response = pb.PrefillResponse(
                request_id=req.request_id,
                cache_handle=cache_handle,
                prefill_duration_ms=duration_ms,
            )

            for token in generated_tokens:
                response.generated_tokens.append(
                    pb.Token(
                        id=token["id"],
                        text=token["text"],
                        logprob=token.get("logprob", 0.0),
                        special=token.get("special", False),
                    )
                )

            responses.append(response)

        logger.info(
            f"Prefill 完成: batch_id={request.batch_id}, "
            f"total_tokens={total_tokens}"
        )

        return pb.BatchPrefillResponse(
            batch_id=request.batch_id,
            responses=responses,
        )

    # ================================================================
    # Decode - 批量解码
    # ================================================================

    def Decode(
        self, request: pb.BatchDecodeRequest, context
    ) -> pb.BatchDecodeResponse:
        """
        批量解码

        对 batch 中所有活跃请求执行一次 decode step，
        每个请求生成一个 token。
        """
        logger.debug(
            f"收到 Decode 请求: batch_id={request.batch_id}, "
            f"requests={len(request.requests)}"
        )

        responses = []

        for req in request.requests:
            params = req.params
            max_new_tokens = params.max_new_tokens if params else 100
            temperature = params.temperature if params else 1.0
            top_p = params.top_p if params else 1.0
            top_k = params.top_k if params else 0
            do_sample = params.do_sample if params else False

            # 调用引擎
            token_dict, finish_reason, duration_ms = self.engine.decode(
                request_id=req.request_id,
                cache_handle=req.cache_handle,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )

            # 构建响应
            response = pb.DecodeResponse(
                request_id=req.request_id,
                decode_duration_ms=duration_ms,
            )

            # 设置 finish_reason
            if finish_reason == "eos_token":
                response.finish_reason = pb.FinishReason.EOS_TOKEN
            elif finish_reason == "length":
                response.finish_reason = pb.FinishReason.MAX_TOKENS
            elif finish_reason == "stop_sequence":
                response.finish_reason = pb.FinishReason.STOP_SEQUENCE
            elif finish_reason == "error":
                response.finish_reason = pb.FinishReason.ERROR_REASON
            else:
                response.finish_reason = pb.FinishReason.NONE

            # 设置 token
            if token_dict:
                response.generated_token.CopyFrom(
                    pb.Token(
                        id=token_dict["id"],
                        text=token_dict["text"],
                        logprob=token_dict.get("logprob", 0.0),
                        special=token_dict.get("special", False),
                    )
                )

            responses.append(response)

        return pb.BatchDecodeResponse(
            batch_id=request.batch_id,
            responses=responses,
        )

    # ================================================================
    # ClearCache - 清理 KV Cache
    # ================================================================

    def ClearCache(
        self, request: pb.ClearCacheRequest, context
    ) -> pb.ClearCacheResponse:
        """清理指定的 KV Cache 条目"""
        logger.debug(f"清理缓存: handles={list(request.cache_handles)}")
        success = self.engine.clear_cache(list(request.cache_handles))
        return pb.ClearCacheResponse(success=success)


def serve(host: str = "0.0.0.0", port: int = 50051, max_workers: int = 10):
    """启动 gRPC 服务器"""

    # 创建模型引擎
    engine = ModelEngine(
        model_id=os.environ.get("MODEL_ID", "mock-gpt2"),
        max_sequence_length=int(os.environ.get("MAX_SEQUENCE_LENGTH", "4096")),
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "32")),
        vocab_size=int(os.environ.get("VOCAB_SIZE", "50257")),
        eos_token_id=int(os.environ.get("EOS_TOKEN_ID", "50256")),
        device=os.environ.get("DEVICE", "cpu"),
    )

    # 创建 gRPC 服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )

    # 注册服务
    servicer = TextGenerationServicer(engine)
    rpc.add_TextGenerationServiceServicer_to_server(servicer, server)

    # 监听
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)

    # 启动
    server.start()
    logger.info(f"=== Light TGI Model Server 启动 ===")
    logger.info(f"监听地址: {addr}")
    logger.info(f"模型: {engine.model_id}")
    logger.info(f"设备: {engine.device}")
    logger.info("按 Ctrl+C 停止服务")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("收到关闭信号，正在优雅退出...")
        server.stop(grace=5)
        logger.info("服务已停止")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Light TGI Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=50051, help="监听端口")
    parser.add_argument("--workers", type=int, default=10, help="工作线程数")
    args = parser.parse_args()

    serve(host=args.host, port=args.port, max_workers=args.workers)
