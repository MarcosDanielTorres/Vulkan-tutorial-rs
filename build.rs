use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    // Locate the Vulkan SDK using an environment variable.
    let vulkan_sdk = env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not set");
    let glsl_compiler;
    if cfg!(target_os = "windows") {
        glsl_compiler = Path::new(&vulkan_sdk).join("Bin/glslc.exe");
    } else if cfg!(all(
        unix,
        not(target_os = "android"),
        not(target_os = "macos")
    )) {
        glsl_compiler = Path::new(&vulkan_sdk).join("x86_64/bin/glslc");
    } else {
        glsl_compiler = Path::new("/usr/local/bin/glslc").to_path_buf();
    }

    // Define the paths to your shader files and output directory.
    let shader_paths = ["src/shaders/shader.vert", "src/shaders/shader.frag"];
    let output_dir = Path::new(&env::var("OUT_DIR").unwrap()).join("shaders");

    // Create the output directory if it doesn't exist.
    std::fs::create_dir_all(&output_dir).expect("Failed to create shader output directory");

    for shader_path in &shader_paths {
        let output_path = output_dir.join(
            Path::new(shader_path)
                .file_name()
                .expect("Failed to extract file name"),
        );

        // Compile each shader using glslc.
        Command::new(&glsl_compiler)
            .args([shader_path, "-o", output_path.to_str().unwrap()])
            .status()
            .expect("Failed to compile shader");
    }

    println!("cargo:rerun-if-changed=src/shaders");
}
