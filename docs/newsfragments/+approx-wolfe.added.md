`Lbfgs::approx_wolfe` is an opt-in approximate Wolfe test for the rounding floor.
A positive value accepts a trial within that fraction of `abs(f)` of the start of a strong Wolfe search when the strong curvature test holds.
Zero, the default, keeps the strict search.
The eOn cap does not consult it. `LbfgsStep::Wolfe` and `Lbfgs::minimize_box` do.
