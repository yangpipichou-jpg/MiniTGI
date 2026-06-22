"""
使用 curl 测试 Router 的 HTTP API

前置条件：
1. Python Model Server 已启动: python model_server/grpc_server.py
2. Rust Router 已编译启动: cargo run --release
"""

import requests
import json
import sys


def test_generate(host="localhost", port=3000):
    """测试 /generate 端点"""
    url = f"http://{host}:{port}/generate"

    payload = {
        "inputs": "What is the capital of France?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "do_sample": True,
        },
    }

    print(f"发送请求到 {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 60)

    try:
        # 使用 stream=True 接收 SSE
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=30,
        )

        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print("-" * 60)
        print("SSE 流式响应:")

        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    try:
                        data = json.loads(data_str)
                        token = data.get("token", {})
                        text = token.get("text", "")
                        generated = data.get("generated_text", "")
                        details = data.get("details")

                        if details:
                            print(f"\n[完成] {details}")
                        else:
                            print(text, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"\n[原始数据] {data_str}")

        print("\n" + "-" * 60)
        print("请求完成!")

    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 {url}，请确保 Router 已启动")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("错误: 请求超时")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def test_concurrent():
    """测试并发请求和过载保护"""
    import concurrent.futures

    url = f"http://localhost:3000/generate"

    def send_request(i):
        payload = {
            "inputs": f"Request number {i}",
            "parameters": {"max_new_tokens": 20},
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            return i, resp.status_code
        except Exception as e:
            return i, str(e)

    print("=" * 60)
    print("并发测试: 发送 20 个并发请求")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(send_request, i) for i in range(20)]
        for future in concurrent.futures.as_completed(futures):
            i, result = future.result()
            print(f"  请求 #{i}: {result}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--concurrent", action="store_true", help="并发测试")
    args = parser.parse_args()

    if args.concurrent:
        test_concurrent()
    else:
        test_generate(args.host, args.port)
