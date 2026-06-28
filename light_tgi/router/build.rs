fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 设置 protoc 路径（使用项目内的 protoc.exe）
    let protoc_path = std::path::Path::new("../protoc.exe");
    if protoc_path.exists() {
        std::env::set_var("PROTOC", protoc_path.canonicalize()?);
    }

    // 编译 protobuf 定义
    tonic_build::configure()
        .build_server(false)  // Router 只需要 client 端
        .build_client(true)
        .compile_protos(
            &["../proto/generation.proto"],
            &["../proto"],
        )?;
    Ok(())
}
