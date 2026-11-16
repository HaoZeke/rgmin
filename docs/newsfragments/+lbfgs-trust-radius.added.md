`Solver::set_lbfgs_trust_radius` truncates the L-BFGS ray and updates the radius by Burdakov's Algorithm 1.
It is off by default. `set_lbfgs_trust` stays the eOn Newton-model radius.
The capped step stays the default until the radius is on, and the eOn fast path does not skip it.
C: `rgmin_solver_set_lbfgs_trust` and `rgmin_solver_lbfgs_trust_radius`, ABI minor 45.
