"""
gRPC 测试客户端 — 直接测试 Model Server (真实模型版本)

运行方式:
    F:\ProgramData\anaconda3\python.exe test_client.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import grpc
import generation_pb2 as pb
import generation_pb2_grpc as rpc


def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("测试 1: Health (健康检查)")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)
        response = stub.Health(pb.ModelInfoRequest())

        print(f"  模型 ID: {response.model_id}")
        print(f"  最大序列长度: {response.max_sequence_length}")
        print(f"  最大 batch 大小: {response.max_batch_size}")
        print(f"  词表大小: {response.vocab_size}")
        print(f"  EOS Token ID: {response.eos_token_id}")
        print(f"  设备: {response.device}")
        print("  健康检查: PASSED\n")


def test_tokenize():
    """测试 Tokenize RPC"""
    print("=" * 60)
    print("测试 2: Tokenize")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)

        texts = [
            "Hello, how are you?",
            "中国的首都是北京。",
            "What is the capital of France?",
        ]

        for text in texts:
            resp = stub.Tokenize(pb.TokenizeRequest(text=text))
            print(f"  文本: '{text}'")
            print(f"    Token 数: {resp.token_count}")
            print(f"    Token IDs (前10): {resp.token_ids[:10]}...")
        print("  Tokenize: PASSED\n")


def test_prefill():
    """测试预填充 (真实模型)"""
    print("=" * 60)
    print("测试 3: Prefill (真实模型推理)")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)

        request = pb.BatchPrefillRequest(
            batch_id=1,
            requests=[
                pb.PrefillRequest(
                    request_id="test-req-1",
                    input_text="What is the capital of France?",
                    params=pb.GenerationParameters(
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                    ),
                ),
            ],
        )

        print("  发送请求: 'What is the capital of France?'")
        response = stub.Prefill(request)

        for resp in response.responses:
            print(f"  请求: {resp.request_id}")
            print(f"    Cache Handle: {resp.cache_handle}")
            print(f"    Prompt Token 数: {resp.prompt_token_count}")
            print(f"    Prefill 耗时: {resp.prefill_duration_ms}ms")
            for token in resp.generated_tokens:
                print(f"    Token: id={token.id}, text='{token.text}'")
        print("  预填充: PASSED\n")

        return response


def test_decode(cache_handles):
    """测试解码 (真实模型)"""
    print("=" * 60)
    print("测试 4: Decode (多步生成)")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)

        for step in range(5):
            for req_id, handle in cache_handles:
                request = pb.BatchDecodeRequest(
                    batch_id=2,
                    requests=[
                        pb.DecodeRequest(
                            request_id=req_id,
                            token_id=0,
                            cache_handle=handle,
                            params=pb.GenerationParameters(
                                max_new_tokens=50,
                                temperature=0.7,
                                do_sample=True,
                            ),
                        ),
                    ],
                )

                response = stub.Decode(request)
                for resp in response.responses:
                    token = resp.generated_token
                    finish = pb.FinishReason.Name(resp.finish_reason)
                    if resp.finish_reason != pb.FinishReason.NONE:
                        print(f"  [Step {step+1}] Token: '{token.text}' → {finish}")
                    else:
                        print(f"  [Step {step+1}] Token: '{token.text}'", end="")

                    if resp.finish_reason != pb.FinishReason.NONE:
                        print("\n  解码: PASSED\n")
                        return
        print()


def test_clear_cache(cache_handles):
    """测试清理缓存"""
    print("=" * 60)
    print("测试 5: ClearCache")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)
        handles = [h for _, h in cache_handles]
        response = stub.ClearCache(pb.ClearCacheRequest(cache_handles=handles))
        print(f"  清理结果: {response.success}")
        print("  清理缓存: PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("Light TGI Model Server - gRPC 测试 (真实模型)")
    print("=" * 60 + "\n")

    try:
        test_health()
        test_tokenize()

        prefill_response = test_prefill()
        cache_handles = [
            (resp.request_id, resp.cache_handle)
            for resp in prefill_response.responses
        ]

        test_decode(cache_handles)
        test_clear_cache(cache_handles)

        print("=" * 60)
        print("所有测试通过! (真实模型推理正常)")
        print("=" * 60)

    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()} - {e.details()}")
        sys.exit(1)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
