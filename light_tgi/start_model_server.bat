@echo off
REM ============================================================
REM Light TGI Model Server Startup Script (Windows GPU)
REM
REM Python: F:\ProgramData\anaconda3\python.exe
REM Model: Qwen/Qwen2.5-7B-Instruct (7B params, GPU recommended)
REM Device: CUDA (GPU)
REM ============================================================

set PYTHON=F:\ProgramData\anaconda3\python.exe

echo ============================================================
echo Light TGI Model Server (GPU)
echo ============================================================
echo Python: %PYTHON%
echo.

REM Check dependencies (includes CUDA info)
echo [1/3] Checking dependencies (may take a moment on first PyTorch load)...
%PYTHON% -c "import sys; sys.stdout.write('Loading PyTorch... '); sys.stdout.flush(); import torch; print('OK'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); import transformers; import grpc; print('All dependencies OK')"
if errorlevel 1 (
    echo Dependency check failed. Installing...
    %PYTHON% -m pip install torch transformers accelerate grpcio grpcio-tools protobuf sentencepiece tiktoken
    echo Re-checking dependencies...
    %PYTHON% -c "import torch; import transformers; import grpc; print('All dependencies OK')"
    if errorlevel 1 (
        echo FATAL: Dependency installation failed!
        pause
        exit /b 1
    )
)

REM Compile proto
echo [2/3] Compiling proto files...
cd /d "%~dp0model_server"
%PYTHON% generate_proto.py
if errorlevel 1 (
    echo Proto compilation failed!
    pause
    exit /b 1
)

REM Start server
echo [3/3] Starting gRPC server...
echo.
echo Model: Qwen/Qwen2.5-7B-Instruct
echo Device: CUDA (GPU)
echo Dtype: float16
echo Port: 50051
echo.
echo First run will download the model from HuggingFace (~15GB), please wait...
echo Model loading takes ~30-60 seconds depending on GPU and network...
echo.

REM Set environment variables
set DEVICE=cuda
set DTYPE=float16
set MODEL_ID=Qwen/Qwen2.5-7B-Instruct
set MAX_BATCH_SIZE=16
set MAX_SEQUENCE_LENGTH=4096

%PYTHON% grpc_server.py --host 0.0.0.0 --port 50051 --model-id %MODEL_ID% --device cuda --dtype float16 --workers 10

pause
