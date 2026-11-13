`SuppliedSearch` and `rgmin_linesearch` take a caller-supplied energy and gradient.
The search does not call an objective. Its trials match `LineSearch::search`.
ABI minor 37.
