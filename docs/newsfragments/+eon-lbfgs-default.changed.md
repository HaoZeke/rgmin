The default L-BFGS step is the eOn line-search-free cap.
The first step probes curvature over 0.01.
A proposal at least as long as the cap clears the memory and steps along `H0` times the force.
`LbfgsStep::Scaled` keeps the shorten-and-retain rule.
FIRE 1.0 treats a zero power as downhill.
