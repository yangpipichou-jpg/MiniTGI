@echo off
REM ============================================================
REM Light TGI Router 启动脚本 (Windows)
REM
REM 前提: Model Server 已启动 (python grpc_server.py)
REM ============================================================

echo ============================================================
echo Light TGI Router (Rust)
echo ============================================================
echo.

cd /d "%~dp0router"

echo 编译 Rust Router...
cargo build --release
if errorlevel 1 (
    echo 编译失败!
    pause
    exit /b 1
)

echo.
echo 启动 Router...
echo HTTP: http://localhost:3000
echo gRPC: http://localhost:50051
echo.

set RUST_LOG=light_tgi=info
set MODEL_SERVER_HOST=127.0.0.1
set MODEL_SERVER_PORT=50051

cargo run --release

pause
