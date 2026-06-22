fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 编译 protobuf 定义
    tonic_build::configure()
        .build_server(false)  // Router 只需要 client 端
        .build_client(true)
        .compile(
            &["../proto/generation.proto"],
            &["../proto"],
        )?;
    Ok(())
}
