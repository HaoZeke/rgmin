`ManifoldKind::Product` is the weighted Riemannian product of other kinds, with the metric `g = sum_i w_i g_i` for weights `w_i > 0`.
`ManifoldKind::product` builds it from `ProductFactor` values, each a kind, a packed length and a weight, and `ManifoldKind::power` is manopt's `powermanifold`; `Solver::set_product` and `Solver::set_power` select them.
Projection, retraction, transport, exponential, logarithm and parallel transport act per factor, a factor's gradient and Hessian are divided by its weight, and the distance is `sqrt(sum_i w_i d_i^2)`.
A session steps a weighted product in the chart `z_i = sqrt(w_i) x_i`, so steepest descent, L-BFGS and the Newton-CG trust region run on it unchanged, and the iterates do not depend on the unit of a factor whose weight carries that unit.
The exponential preconditioner, the Lindh model, `set_project_rigid` and the HiGHS box read raw coordinates and return `Error::ProductChart` on such a product.
`MwRigid` and `RigidQuotient` factors project with the session's masses and periodic flag.
`explanation/product-manifold` derives the geometry and compares it with manopt, Manopt.jl and Manifolds.jl, pymanopt and geomstats.
