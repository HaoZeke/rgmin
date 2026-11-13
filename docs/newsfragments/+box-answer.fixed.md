`Lbfgs::minimize_box` honors `ftol`, the same relative-decrease stop as `minimize`.
A short gradient, a NaN bound, an inverted bound, or a bound of the wrong length ends the call at `+inf` instead of panicking.
