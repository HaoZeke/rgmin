# Contributing to rgmin

Report a bug or ask a question on the [issue tracker](https://github.com/OmniPotentRPC/rgmin/issues).
Report a vulnerability as the [security policy](SECURITY.md) says, not in a public issue.
Everyone who takes part follows the [code of conduct](CODE_OF_CONDUCT.md), in issues and pull requests alike.

Pull requests go to the `develop` branch.
Before you open one, run the checks that continuous integration (CI) runs with Rust 1.88.0:

```bash
rustup toolchain install 1.88.0 --profile minimal -c rustfmt
python3 -m pip install numpy scipy
cargo +1.88.0 fmt --all -- --check
cargo +1.88.0 test --locked
cargo +1.88.0 test --locked --features capi
cargo +1.88.0 test --locked --features highs
cargo +1.88.0 test --locked --all-features
```

A change that users can see adds an entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

The [contributing page](docs/orgmode/contributing/index.org) of the docs covers the rest: the C library check, the docs build, citations and the formula numbers the code follows.
