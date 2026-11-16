`ManifoldKind::sphere_within(basis)` is the unit sphere of the span of the rows of `basis`, and `ManifoldKind::sphere_excluding(basis)` the unit sphere orthogonal to them, such as a unit normal that leaves a centre of mass in place.
Both are `SubspaceSphere`, with the projection `P_U - x x^T`, the metric-projection retraction, and the sphere's exponential, logarithm, distance, parallel transport and Hessian.
`Error::SubspaceBasis` reports an empty, non-finite or dependent basis, or excluded rows that span the whole space.
