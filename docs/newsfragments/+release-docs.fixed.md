The user docs describe the 0.6.0 code.
L-BFGS takes the eOn step by default, with a per-atom cap and no line search, and `LbfgsStep::Wolfe` opts into the strong Wolfe search.
A session's line search defaults to strong Wolfe for the methods that use one.
The install guide covers crates.io, git, cargo-c, Meson, both Python packages and the LAMMPS plugin, and its commands run as written.
The README, the reference pages and the contributing guide give the current ABI minor, methods, manifold kinds and CI rows.
The LAMMPS benchmark table names the commit and crate version it was measured at.
The Sphinx build passes with warnings as errors.
