"""
编译 protobuf 定义，生成 Python gRPC 代码

运行方式：
    python generate_proto.py

或使用 grpcio-tools：
    pip install grpcio-tools
    python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. ../proto/generation.proto
"""

import os
import sys


def generate():
    """使用 grpc_tools 编译 proto 文件"""
    proto_dir = os.path.join(os.path.dirname(__file__), "..", "proto")
    proto_file = os.path.join(proto_dir, "generation.proto")
    output_dir = os.path.dirname(__file__)

    # 构建命令
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file,
    ]

    print(f"编译 proto 文件: {proto_file}")
    print(f"输出目录: {output_dir}")

    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"编译失败:\n{result.stderr}")
        sys.exit(1)

    print("编译成功!")
    print(f"生成文件:")
    for f in os.listdir(output_dir):
        if f.endswith("_pb2.py") or f.endswith("_pb2_grpc.py"):
            print(f"  - {f}")


if __name__ == "__main__":
    generate()
