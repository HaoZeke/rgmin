A steady `Solver::step` for L-BFGS (`Accept::Energy`) and FIRE 2.0
allocates nothing outside the objective. The first step at a length
still builds the scratch. The flat FIRE velocity update, mix and
displacement each run as one pass.
