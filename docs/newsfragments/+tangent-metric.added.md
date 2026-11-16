`ProductFactor::with_metric` gives a factor a caller's `TangentMetric` `G_x`, self-adjoint and positive definite on each tangent space, so that its part of the product metric is `w g(u, G_x v)`.
`egrad2rgrad` returns the factor's gradient times `(w G_x)^{-1}`, and `minimize_rtr` then has the iterates of manopt's truncated CG preconditioned by `(w G_x)^{-1}`.
Such a product has no exponential, logarithm, distance or parallel transport, and sessions refuse it with `Error::ProductTangentMetric`.
