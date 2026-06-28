@echo off
REM ============================================================
REM Light TGI Router Startup Script (Windows GPU)
REM
REM Prerequisite: Model Server must be running (python grpc_server.py --device cuda)
REM ============================================================

echo ============================================================
echo Light TGI Router (Rust) - GPU
echo ============================================================
echo.

cd /d "%~dp0router"

echo Building Rust Router...
cargo build --release
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Starting Router...
echo HTTP: http://localhost:3000
echo gRPC: http://localhost:50051
echo.

REM Router configuration (GPU optimized)
set RUST_LOG=light_tgi=info
set MODEL_SERVER_HOST=127.0.0.1
set MODEL_SERVER_PORT=50051

REM Higher concurrency and batch for GPU inference
set MAX_CONCURRENT_REQUESTS=128
set MAX_BATCH_SIZE=16
set MAX_BATCH_PREFILL_TOKENS=4096
set MAX_BATCH_TOTAL_TOKENS=16384
set MAX_WAITING_TOKENS=20
set WAITING_SERVED_RATIO=1.2
set MAX_INPUT_LENGTH=4096
set MAX_TOTAL_TOKENS=8192

cargo run --release

pause
