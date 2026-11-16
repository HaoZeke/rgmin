`rgmin.optimize` is a drop-in for `ase.optimize` on the maturin session.
`BFGS`, `LBFGS`, `LBFGSLineSearch`, `FIRE` and `FIRE2` take ASE's constructor arguments and drive `self.optimizable`, so cell filters and `FixAtoms`, `FixedLine`, `FixCartesian` and `FixedPlane` work.
The other gradient methods of `Session` have ASE-style classes. Importing `rgmin` does not import ASE.
`Session` takes `h0`, `maxmove`, `preset`, `abc`, `accept` and `istep`, and can `push_pair`, `forget` and report `pair_count` and `steps`.
`Adam` takes `learning_rate` (default 0.02) as its step scale.
A restart file replays the recorded evaluations into a fresh session.
