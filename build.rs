fn main() {
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rustc-check-cfg=cfg(rgmin_has_libkrylov)");
    #[cfg(feature = "libkrylov")]
    probe_libkrylov();
}

#[cfg(feature = "libkrylov")]
struct LibkrylovProbe {
    includes: Vec<std::path::PathBuf>,
    link_paths: Vec<std::path::PathBuf>,
    link_libs: Vec<String>,
}

#[cfg(feature = "libkrylov")]
fn probe_libkrylov() {
    println!("cargo:rerun-if-changed=src/libkrylov_shim.c");
    println!("cargo:rerun-if-env-changed=LIBKRYLOV_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");

    let Some(cfg) = discover_libkrylov() else {
        println!(
            "cargo:warning=libkrylov feature on, libkrylov not found; EigensolverKind::Libkrylov stays EigenUnavailable"
        );
        return;
    };
    let mut build = cc::Build::new();
    build.file("src/libkrylov_shim.c");
    build.warnings(false);
    for inc in &cfg.includes {
        build.include(inc);
    }
    match build.try_compile("rgmin_libkrylov_shim") {
        Ok(()) => {
            for path in &cfg.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            for lib in &cfg.link_libs {
                println!("cargo:rustc-link-lib={lib}");
            }
            println!("cargo:rustc-cfg=rgmin_has_libkrylov");
        }
        Err(err) => {
            println!(
                "cargo:warning=libkrylov shim did not compile ({err}); EigensolverKind::Libkrylov stays EigenUnavailable"
            );
        }
    }
}

#[cfg(feature = "libkrylov")]
fn discover_libkrylov() -> Option<LibkrylovProbe> {
    for name in ["krylov", "libkrylov"] {
        if let Ok(lib) = pkg_config::Config::new()
            .cargo_metadata(false)
            .probe(name)
        {
            let mut link_paths = lib.link_paths;
            let mut link_libs = lib.libs;
            if !link_libs.iter().any(|n| n == "openblas" || n == "blas") {
                if let Some((path, name)) = discover_openblas() {
                    if !link_paths.iter().any(|p| p == &path) {
                        link_paths.push(path);
                    }
                    link_libs.push(name);
                }
            }
            if !link_libs.iter().any(|n| n == "gfortran") {
                link_libs.push("gfortran".into());
            }
            return Some(LibkrylovProbe {
                includes: lib.include_paths,
                link_paths,
                link_libs,
            });
        }
    }
    if let Ok(dir) = std::env::var("LIBKRYLOV_DIR") {
        if let Some(probe) = probe_libkrylov_prefix(std::path::Path::new(&dir)) {
            return Some(probe);
        }
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        if let Some(probe) = probe_libkrylov_prefix(std::path::Path::new(&prefix)) {
            return Some(probe);
        }
    }
    probe_libkrylov_prefix(std::path::Path::new("/usr"))
}

#[cfg(feature = "libkrylov")]
fn libkrylov_header_in(dir: &std::path::Path) -> bool {
    dir.join("ckrylov.h").is_file() || dir.join("krylov").join("ckrylov.h").is_file()
}

#[cfg(feature = "libkrylov")]
fn probe_libkrylov_prefix(prefix: &std::path::Path) -> Option<LibkrylovProbe> {
    let include = prefix.join("include");
    let lib = prefix.join("lib");
    let has_lib = lib.join("libkrylov.a").is_file()
        || lib.join("libkrylov.so").is_file()
        || lib.join("libkrylov.dylib").is_file();
    if !has_lib || !libkrylov_header_in(&include) {
        return None;
    }
    let mut link_paths = vec![lib];
    let mut link_libs = vec!["krylov".into(), "gfortran".into()];
    if let Some((path, name)) = discover_openblas() {
        if !link_paths.iter().any(|p| p == &path) {
            link_paths.push(path);
        }
        link_libs.push(name);
    } else {
        link_libs.push("lapack".into());
        link_libs.push("blas".into());
    }
    Some(LibkrylovProbe {
        includes: vec![include],
        link_paths,
        link_libs,
    })
}

#[cfg(feature = "libkrylov")]
fn discover_openblas() -> Option<(std::path::PathBuf, String)> {
    for root in [
        std::env::var_os("OPENBLAS_DIR").map(std::path::PathBuf::from),
        std::env::var_os("CONDA_PREFIX").map(std::path::PathBuf::from),
    ]
    .into_iter()
    .flatten()
    {
        let lib = root.join("lib");
        if lib.join("libopenblas.so").is_file() || lib.join("libopenblas.a").is_file() {
            return Some((lib, "openblas".into()));
        }
    }
    let usr = std::path::Path::new("/usr/lib");
    if usr.join("libopenblas.so").is_file() {
        return Some((usr.to_path_buf(), "openblas".into()));
    }
    None
}
