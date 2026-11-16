`Manifold` gains manopt's `inner`, `norm`, `ehess2rhess`, `exp`, `log`, `dist`, `parallel_transport`, `parallel_transport_along`, `transport_is_isometric` and `typical_dist`.
Their defaults keep every kind as it was: the Euclidean metric, and `None` where no closed form is supplied.
Every kind but the multinomial and rigid kinds supplies its Riemannian Hessian, the sphere family, SO(3), SE(3), `Positive` and the linear kinds also their geodesics, and `Positive`, `Spd` and the doubly stochastic kinds expose the metrics their `egrad2rgrad` already used, as does the trait implementation of `RiemannianManifold`.
`reference/product-manifold` lists what each kind supplies.
