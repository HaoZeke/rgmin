The FIRE fill step cap folds atom distances with iterators instead of an indexed `f64::max` loop.
rustc 1.93 and 1.94 (LLVM 21) segfault in LoopVectorizePass on that indexed loop, so `cargo build --release` failed there.
A non-positive or non-finite cap still leaves the trial alone.
The rescaled coordinates match the indexed loops bit for bit.
