`LbfgsStep::Capped` drops the L-BFGS line search and shortens each two-loop step to a per-atom cap.
The initial inverse Hessian stays the Nocedal-Wright scale `(s·y)/(y·y)`.
`Solver::set_lbfgs_step` installs the rule, and `rgmin_solver_set_lbfgs_capped` is the C entry.
`abi_minor` 33.
