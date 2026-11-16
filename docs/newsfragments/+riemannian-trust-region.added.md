`rtr::minimize_rtr` is the Riemannian trust region of Absil, Baker and Gallivan with Steihaug-Toint truncated CG, measured in `Manifold::inner`.
Its Hessian action is `ehess2rhess`, Weingarten term included, where the kind supplies one, and the projected Euclidean action otherwise.
`RtrOptions` takes the defaults of manopt's `trustregions`, and `RtrMin` returns the point, the value, the gradient norm and the outer, accepted and inner iteration counts.
`rtr::truncated_cg_riemannian` is one inner solve in the manifold metric, for a caller that accepts or rejects steps itself with `RtrRadius`.
`truncated_cg_projected` keeps the Euclidean inner product.
