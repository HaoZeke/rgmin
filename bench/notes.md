# What changed, and what did not

Five serial repeats. Medians are the optimizer and the potential
together. The before column is the session loop at
`2e152a9447343190a763d84403d7e82913b49e6e`. The after column is
`Solver::run_fill` for the eOn cap, FIRE, and the new conjugate
gradient. eOn is `bench/eon_ref.cpp` from TheochemUI/eOn `579738a`.
Rows are `bench/raw/`.

| problem | arm | before ms | after ms | eOn ms | evals |
|---|---|---:|---:|---:|---:|
| Rosenbrock2 | L-BFGS m20 | 0.0584 | 0.0185 | 0.0211 | 66 |
| Rosenbrock10 | L-BFGS m20 | 0.1557 | 0.0576 | 0.0736 | 141 vs 149 |
| LJ7 | L-BFGS m20 | 0.0536 | 0.0198 | 0.0366 | 33 |
| Rosenbrock2 | FIRE | 0.9760 | 0.1548 | 0.2045 | 2714 |
| Rosenbrock10 | FIRE | 2.0701 | 0.4410 | 0.4765 | 4183 |
| LJ7 | FIRE | 0.0602 | 0.0190 | 0.0396 | 82 |
| Rosenbrock2 | CG |  | 0.0043 | 0.0084 | 87 |
| LJ7 | CG |  | 0.0070 | 0.0188 | 37 |

The L-BFGS and FIRE evaluation counts on these rows match the
starting commit. CG matches eOn on Rosenbrock 2 and LJ7.

## Per-step cost

The eOn step cloned every stored pair at the start of the iteration
so a non-finite trial could be undone. With memory 20 that is forty
vectors before the two-loop allocates its own residual and
coefficient buffer. The undo log moves a pair aside and puts it
back. A successful step recycles the buffers. The two-loop writes
into a workspace and calls the same dot and axpy as `direction_h0`.

`Solver::run_fill` keeps the gradient in one buffer. Euclidean FIRE
1.0 and the eOn cap add the displacement in place. `Solver::step`
still returns a report that owns the coordinates and the gradient.

A counting allocator on `Solver::run` (report vectors suppressed)
measured about 2.3 heap allocations per L-BFGS step and 1.0 per FIRE
step on Rosenbrock 2, almost all of them the gradient returned by
the objective. `run_fill` removes that return.

## Scaling

For `H0 = gamma I` the secant `H0 y = s` is one equation in `n`
unknowns. The least-squares `gamma` is `(s · y) / (y · y)`:

```
d/d(gamma) ||gamma y - s||^2 = 2 (gamma y - s) · y = 0
gamma = (s · y) / (y · y)
```

SymPy reduces that quotient to `s·y/y·y`. A diagonal quadratic
`H = diag(2, 8)` gives a finite-difference curvature
`((g(x+δu) - g) · u) / δ` equal to the Rayleigh quotient `uᵀ H u`
(`5.840000000000134` against `5.840000000000001`). The probe
`H0 = 1/C` is that reciprocal. This is the scale already used
(Liu and Nocedal, Mathematical Programming 45, 1989,
<https://doi.org/10.1007/BF01589116>; Nocedal and Wright,
<https://doi.org/10.1007/978-0-387-40065-5>).

PySR 2.7.2, six iterations, features `(s·y, y·y, ||g||)` and target
`(s·y)/(y·y)`, returned `sy*sy/(sy*yy)`. That is the same scale.
Held-out mean absolute error was `3.7e-17` against an exact
Liu-Nocedal error of 0. It was not adopted as a new default.

## Not adopted

Goswami's dimer benchmarks compare CG and L-BFGS as rotation
optimizers and keep L-BFGS for the translational step
(<https://arxiv.org/abs/2505.13621>,
<https://doi.org/10.1063/5.0283639>;
<https://arxiv.org/html/2505.12519>). They do not change the
line-search-free cap used here.

A left-to-right slice product in the two-loop changed Rosenbrock 10
from 141 evaluations to 149 and moved the Wolfe LJ38 run. It was
dropped. The kept two-loop uses `vecops`.

CG_DESCENT and limited-memory CG (Hager and Zhang, SIAM J. Optim. 23,
2013, <https://doi.org/10.1137/120898097>; the 2005 method is the
reference listed in
<https://github.com/iskolbin/cg_descent/blob/master/cg_descent.h>)
need a Wolfe line search. Strong Wolfe stays opt-in, so neither
replaced the capped step.

FIRE 2.0 and ABC-FIRE (Guénolé et al., Computational Materials
Science 175, 2020, <https://doi.org/10.1016/j.commatsci.2020.109584>;
Echeverri Restrepo and Andric, Computational Materials Science 218,
2023, <https://doi.org/10.1016/j.commatsci.2022.111978>) are already
in the session. They are not the eOn FIRE default, which is the
Bitzek 2006 integrator
(<https://doi.org/10.1103/PhysRevLett.97.170201>) with time step 1.

On Rosenbrock 10 the new CG reaches the tolerance in 297 evaluations
where eOn uses 883. LJ13 is 179 against 175, and LJ38 is 543 against
397 and stops in a lower basin. The Rosenbrock 2 and LJ7 paths match
eOn to the reported digits. The difference is rounding in the
recurrence, not a different beta. No restart rule was changed.

The line-search-free cap remains the default.
