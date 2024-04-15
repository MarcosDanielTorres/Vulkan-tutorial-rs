use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Locate the Vulkan SDK using an environment variable.
    let vulkan_sdk = env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not set");
    let glsl_compiler = determine_glsl_compiler(&vulkan_sdk);

    // Define the path to your shader files and output directory.
    let shaders_dir = Path::new("src\\shaders");
    let output_dir = Path::new(&env::var("OUT_DIR").unwrap()).join("shaders");

    // Create the output directory if it doesn't exist.
    match fs::create_dir_all(&output_dir) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to create shader output directory: {}", e);
            std::process::exit(1);
        }
    };

    // Traverse shaders directory and handle each file
    process_directory(&shaders_dir, &glsl_compiler, &output_dir);
}

fn process_directory(dir: &Path, glsl_compiler: &PathBuf, output_dir: &PathBuf) {
    if dir.is_dir() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                // Recursively process each directory
                process_directory(&path, glsl_compiler, output_dir);
            } else {
                // Print path for cargo to watch and compile the shader
                println!("cargo:rerun-if-changed={}", path.display());
                let output_path = output_dir.join(path.file_name().unwrap());

                // Compile each shader using glslc
                match Command::new(glsl_compiler)
                    .args([path.to_str().unwrap(), "-o", output_path.to_str().unwrap()])
                    .status()
                {
                    Ok(status) => {
                        if !status.success() {
                            eprintln!("Failed to compile shader: {}", path.display());
                            std::process::exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to compile shader: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}

fn determine_glsl_compiler(vulkan_sdk: &str) -> PathBuf {
    if cfg!(target_os = "windows") {
        Path::new(vulkan_sdk).join("Bin/glslc.exe")
    } else if cfg!(all(
        unix,
        not(target_os = "android"),
        not(target_os = "macos")
    )) {
        Path::new(vulkan_sdk).join("x86_64/bin/glslc")
    } else {
        Path::new("/usr/local/bin/glslc").to_path_buf()
    }
}
