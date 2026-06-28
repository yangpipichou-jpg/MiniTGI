"""
Light TGI HTTP API 测试 (真实模型版本)

前置条件:
1. Python Model Server 已启动: start_model_server.bat
2. Rust Router 已编译启动: start_router.bat

用法:
    F:\ProgramData\anaconda3\python.exe test_api.py
    F:\ProgramData\anaconda3\python.exe test_api.py --concurrent
"""

import requests
import json
import sys
import time


def test_generate(host="localhost", port=3000):
    """测试 /generate 端点 (SSE 流式输出)"""
    url = f"http://{host}:{port}/generate"

    payload = {
        "inputs": "What is the capital of France? Answer briefly.",
        "parameters": {
            "max_new_tokens": 80,
            "temperature": 0.7,
            "do_sample": True,
        },
    }

    print(f"POST {url}")
    print(f"Input: {payload['inputs']}")
    print("-" * 60)

    try:
        start = time.time()
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=60,
        )

        print(f"Status: {response.status_code}")
        print("-" * 60)
        print("Output: ", end="", flush=True)

        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    token = data.get("token", {})
                    text = token.get("text", "")
                    details = data.get("details")

                    if details:
                        elapsed = time.time() - start
                        print(f"\n\n[完成] {details['finish_reason']}")
                        print(f"生成 {details['generated_tokens']} tokens")
                        print(f"总耗时: {elapsed:.2f}s")
                    else:
                        print(text, end="", flush=True)
                except json.JSONDecodeError:
                    pass

        print("\n" + "-" * 60)

    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接 {url}，请确保 Router 已启动")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("错误: 请求超时")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def test_concurrent():
    """并发测试"""
    import concurrent.futures

    url = "http://localhost:3000/generate"

    prompts = [
        "What is Python?",
        "Explain machine learning in one sentence.",
        "What is the capital of China?",
        "Who wrote Romeo and Juliet?",
    ]

    def send_request(i, prompt):
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 80, "temperature": 0.7, "do_sample": True},
        }
        try:
            start = time.time()
            resp = requests.post(url, json=payload, timeout=30, stream=True)
            tokens = []
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data = json.loads(line[6:])
                    t = data.get("token", {}).get("text", "")
                    tokens.append(t)
            elapsed = time.time() - start
            return i, resp.status_code, "".join(tokens), elapsed
        except Exception as e:
            return i, 0, str(e), 0

    print("=" * 60)
    print(f"并发测试: {len(prompts)} 个请求")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futures = [ex.submit(send_request, i, p) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futures):
            i, status, text, elapsed = f.result()
            print(f"  #{i}: status={status}, time={elapsed:.2f}s")
            if text and len(text) < 200:
                print(f"       output: {text.strip()[:100]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--concurrent", action="store_true")
    args = parser.parse_args()

    if args.concurrent:
        test_concurrent()
    else:
        test_generate(args.host, args.port)
