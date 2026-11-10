Guénolé FIRE 2.0 no longer allocates on a steady step, and the
recycle pool keeps one length's pair history (`2 * max_pairs`).
A dot at `n >= 65536` is a fixed-chunk pairwise sum on every thread
count. Each two-loop direction is one sweep, with the final negation
folded into the last update.
