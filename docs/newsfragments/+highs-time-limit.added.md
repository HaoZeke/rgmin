Added `HighsOptions::time_limit` and `HighsOptions::with_time_limit`. A value
replaces the time limit of both the projection and the feasible step HiGHS
solves, and `None`, the default, keeps the 0.05 s and 1.0 s limits they had.
The feasible step solve also reports a refused `parallel` or `threads` option
as `Error::Highs` instead of ignoring it.
