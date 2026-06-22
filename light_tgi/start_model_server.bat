@echo off
REM ============================================================
REM Light TGI Model Server 启动脚本 (Windows)
REM
REM Python 环境: F:\ProgramData\anaconda3\python.exe
REM 模型: Qwen/Qwen2.5-1.5B-Instruct (1.5B 参数)
REM ============================================================

set PYTHON=F:\ProgramData\anaconda3\python.exe

echo ============================================================
echo Light TGI Model Server (真实模型版本)
echo ============================================================
echo Python: %PYTHON%
echo.

REM 检查依赖
echo [1/3] 检查依赖...
%PYTHON% -c "import torch; import transformers; import grpc" 2>nul
if errorlevel 1 (
    echo 安装依赖中...
    %PYTHON% -m pip install torch transformers accelerate grpcio grpcio-tools protobuf sentencepiece tiktoken
)

REM 编译 proto
echo [2/3] 编译 proto 文件...
cd /d "%~dp0model_server"
%PYTHON% generate_proto.py
if errorlevel 1 (
    echo proto 编译失败!
    pause
    exit /b 1
)

REM 启动服务
echo [3/3] 启动 gRPC 服务器...
echo.
echo 模型: Qwen/Qwen2.5-1.5B-Instruct
echo 设备: CPU
echo 端口: 50051
echo.
echo 首次运行会从 HuggingFace 下载模型 (~3GB)，请耐心等待...
echo.

%PYTHON% grpc_server.py --host 0.0.0.0 --port 50051 --model-id Qwen/Qwen2.5-1.5B-Instruct --device cpu --workers 4

pause
