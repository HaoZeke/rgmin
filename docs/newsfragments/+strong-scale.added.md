The `strong_scale` example, behind the `par` feature, measures CPU
strong scaling on one fixed FCC Morse crystal.
It runs L-BFGS (m = 10, strong Wolfe) and FIRE 2.0 with the Guenole
2020 preset on 1, 2, 4, ... threads up to the physical cores.
The CSV lists the hardware, threads, N, median wall time, force calls,
iterations, final gradient norm, the largest coordinate deviation from
1 thread, and the parallel efficiency.
CI runs a short smoke of it.
