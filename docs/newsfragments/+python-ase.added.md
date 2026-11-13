The `python/` package builds an ASE `Optimizer` drop-in: `BFGS`, `LBFGS`, and `FIRE` take `atoms`, `trajectory`, `logfile`, `run(fmax, steps)`, and `attach`.
`LBFGS` uses ASE's length control unless `eon_cap=True`.
