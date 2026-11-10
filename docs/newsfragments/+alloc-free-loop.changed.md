The L-BFGS two-loop, its curvature pairs and the line-search trial
reuse their length-n buffers after the first iteration at that length.
A warm `Lbfgs::minimize` iteration allocates only inside the objective.
The arithmetic is unchanged, so a recorded Rosenbrock trajectory matches
bit for bit.
