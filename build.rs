use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=include/rgmin.h");
    println!("cargo:rerun-if-changed=include/dlpack/dlpack.h");
    println!("cargo:rerun-if-changed=Cargo.toml");
    check_public_header();
}

fn check_public_header() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest);
    let header_path = root.join("include/rgmin.h");
    let header = std::fs::read_to_string(&header_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", header_path.display()));
    let dlpack = root.join("include/dlpack/dlpack.h");
    if !dlpack.is_file() {
        panic!(
            "{} is required so cargo cinstall can ship DLPack beside rgmin.h",
            dlpack.display()
        );
    }

    let ver = env!("CARGO_PKG_VERSION");
    let mut parts = ver.split('.');
    let major = parts.next().expect("crate version major");
    let minor = parts.next().expect("crate version minor");
    let patch_raw = parts.next().expect("crate version patch");
    let patch = patch_raw.split(['-', '+']).next().unwrap_or(patch_raw);
    for (name, value) in [
        ("RGMIN_H_MAJOR", major),
        ("RGMIN_H_MINOR", minor),
        ("RGMIN_H_PATCH", patch),
    ] {
        let line = format!("#define {name} {value}\n");
        if !header.contains(&line) {
            panic!("include/rgmin.h must contain `{line}` to match crate version {ver}");
        }
    }
    if header.contains("RGMIN.H_") {
        panic!("include/rgmin.h has a dotted version macro");
    }
    if header.contains("HighsInt") {
        panic!("include/rgmin.h must not name HiGHS HighsInt");
    }
    if header.contains("namespace ") || header.contains("namespace\t") {
        panic!("include/rgmin.h must not open a C++ namespace");
    }
}
