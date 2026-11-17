`minimize_scg`, `minimize_scg_exact`, and `rgmin_minimize_scg` now apply a positive finite `Control::maxmove` to every trial step, including retries after non-finite values.
Bounded steps compare their measured change with the damped model at the shortened step; zero, negative, infinite, and NaN caps remain unbounded.
