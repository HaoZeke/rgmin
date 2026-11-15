`rgmin.optimize` is a drop-in for `ase.optimize`.
`BFGS`, `LBFGS`, `LBFGSLineSearch`, `FIRE` and `FIRE2` take ASE 3.29's constructors and drive `self.optimizable`, so ASE's cell filters and constraints work.
On the same start `BFGS`, `LBFGS` and `FIRE` reproduce ASE's steps, force calls, log lines and trajectory frames.
Restart files replay the recorded evaluations without calling the calculator.
`SR1`, `SR2`, `SteepestDescent`, `BarzilaiBorwein`, `Adam`, the nonlinear CG family, `EonCG`, `NewtonCG` and `NewtonPCG` cover the other session methods.
`howto/ase-migration` lists the differences from ASE, each with a test, and `benches/ase_optimize.py` compares both packages on EMT clusters and slabs, LJ38 and a `FrechetCellFilter` vacancy cell.
