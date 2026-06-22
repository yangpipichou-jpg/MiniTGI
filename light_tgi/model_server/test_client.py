"""
测试客户端 - 直接测试 Model Server 的 gRPC 接口

运行方式：
    1. 先启动 model server:
       python grpc_server.py
    
    2. 再运行测试:
       python test_client.py
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
        print("  健康检查: PASSED\n")


def test_prefill():
    """测试预填充"""
    print("=" * 60)
    print("测试 2: Prefill (预填充)")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)

        # 构造请求
        request = pb.BatchPrefillRequest(
            batch_id=1,
            requests=[
                pb.PrefillRequest(
                    request_id="test-req-1",
                    input_ids=[1, 2, 3, 4, 5],
                    params=pb.GenerationParameters(
                        max_new_tokens=50,
                        temperature=1.0,
                        do_sample=False,
                    ),
                    slot_ids=[0],
                    batch_id=0,
                ),
                pb.PrefillRequest(
                    request_id="test-req-2",
                    input_ids=[10, 20, 30],
                    params=pb.GenerationParameters(
                        max_new_tokens=30,
                        temperature=0.8,
                        do_sample=True,
                    ),
                    slot_ids=[1],
                    batch_id=1,
                ),
            ],
        )

        response = stub.Prefill(request)

        print(f"  Batch ID: {response.batch_id}")
        for resp in response.responses:
            print(f"  请求: {resp.request_id}")
            print(f"    Cache Handle: {resp.cache_handle}")
            print(f"    Prefill 耗时: {resp.prefill_duration_ms}ms")
            for token in resp.generated_tokens:
                print(f"    Token: id={token.id}, text='{token.text}'")
        print("  预填充: PASSED\n")

        return response


def test_decode(cache_handles):
    """测试解码"""
    print("=" * 60)
    print("测试 3: Decode (解码)")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)

        for i, (req_id, handle) in enumerate(cache_handles):
            request = pb.BatchDecodeRequest(
                batch_id=2,
                requests=[
                    pb.DecodeRequest(
                        request_id=req_id,
                        token_id=0,
                        cache_handle=handle,
                        params=pb.GenerationParameters(
                            max_new_tokens=50,
                            temperature=1.0,
                        ),
                    ),
                ],
            )

            response = stub.Decode(request)
            for resp in response.responses:
                token = resp.generated_token
                print(f"  请求: {resp.request_id}")
                print(f"    Token: id={token.id}, text='{token.text}'")
                print(f"    Finish: {pb.FinishReason.Name(resp.finish_reason)}")
                print(f"    耗时: {resp.decode_duration_ms}ms")
        print("  解码: PASSED\n")


def test_clear_cache(cache_handles):
    """测试清理缓存"""
    print("=" * 60)
    print("测试 4: ClearCache (清理缓存)")
    print("=" * 60)

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = rpc.TextGenerationServiceStub(channel)

        handles = [h for _, h in cache_handles]
        response = stub.ClearCache(
            pb.ClearCacheRequest(cache_handles=handles)
        )

        print(f"  清理结果: {response.success}")
        print("  清理缓存: PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("Light TGI Model Server - gRPC 测试")
    print("=" * 60 + "\n")

    try:
        # 1. Health
        test_health()

        # 2. Prefill
        prefill_response = test_prefill()

        # 提取 cache handles
        cache_handles = [
            (resp.request_id, resp.cache_handle)
            for resp in prefill_response.responses
        ]

        # 3. Decode (多步)
        for step in range(3):
            print(f"--- Decode Step {step + 1} ---")
            test_decode(cache_handles)

        # 4. ClearCache
        test_clear_cache(cache_handles)

        print("=" * 60)
        print("所有测试通过!")
        print("=" * 60)

    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()} - {e.details()}")
        sys.exit(1)
    except Exception as e:
        print(f"测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
