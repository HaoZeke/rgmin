`rgmin.optimize.BFGS` starts from ASE's `alpha * I` instead of the identity, `LBFGS` keeps ASE's fixed `H0 = I / alpha` instead of the Nocedal-Wright scale, and `FIRE` is ASE's Euler step with a Euclidean `maxstep` instead of a per-atom cap.
`LBFGS(eon_cap=True)` still selects the crate's default step.
