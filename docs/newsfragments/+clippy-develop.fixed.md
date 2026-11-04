`cargo clippy --all-targets` with warnings denied passes on Rust 1.88.0 and 1.99.1.
It passes for each feature set that CI tests: the defaults, `capi`, `highs`, and all features.
Every fix keeps the solvers' behaviour, and no public signature changes.
