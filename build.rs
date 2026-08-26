fn main() {
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=include/");
    println!("cargo:rerun-if-changed=src/slepc_shim.c");
    if std::env::var("CARGO_FEATURE_SLEPC").is_ok() {
        link_slepc();
    }
}

/// Feature `slepc` links PETSc/SLEPc from the host prefix. The waist
/// does not invent a second PETSc world or read the options database.
fn link_slepc() {
    println!("cargo:rerun-if-env-changed=PETSC_DIR");
    println!("cargo:rerun-if-env-changed=SLEPC_DIR");
    println!("cargo:rerun-if-env-changed=PETSC_ARCH");
    let petsc_dir = std::env::var("PETSC_DIR").unwrap_or_default();
    let slepc_dir = std::env::var("SLEPC_DIR").unwrap_or_default();
    if petsc_dir.is_empty() || slepc_dir.is_empty() {
        panic!(
            "feature \"slepc\" requires PETSC_DIR and SLEPC_DIR \
             (host already lives in PETSc and can supply a Pmat)"
        );
    }
    let arch = std::env::var("PETSC_ARCH").unwrap_or_default();
    let mut includes = Vec::new();
    let mut libs = Vec::new();
    for dir in [&petsc_dir, &slepc_dir] {
        includes.push(format!("{dir}/include"));
        libs.push(format!("{dir}/lib"));
        if !arch.is_empty() {
            includes.push(format!("{dir}/{arch}/include"));
            libs.push(format!("{dir}/{arch}/lib"));
        }
    }
    let mut build = cc::Build::new();
    build.file("src/slepc_shim.c");
    for inc in &includes {
        if std::path::Path::new(inc).is_dir() {
            build.include(inc);
        }
    }
    build.compile("rgmin_slepc_shim");
    for lib in &libs {
        if std::path::Path::new(lib).is_dir() {
            println!("cargo:rustc-link-search=native={lib}");
        }
    }
    println!("cargo:rustc-link-lib=slepc");
    println!("cargo:rustc-link-lib=petsc");
}
