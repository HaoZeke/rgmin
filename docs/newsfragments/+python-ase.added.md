The maturin package under `python/` builds `BFGS`, `LBFGS` and `FIRE` as ASE `Optimizer` subclasses.
`LBFGS` uses ASE's length control unless `eon_cap=True`.
The nanobind `rgmin.ase` classes released in 0.4.0 are unchanged.
