`rtr::minimize_rtr_gauged` calls a gauge after every accepted step: a map to another representation of the same point, such as the pivot of a plane moved along the plane.
The step is accepted on the value before the move, and the moved point is evaluated afresh.
`Solver::run_gauged` calls the gauge before every session step but the first and rebases the session when it moves the point.
