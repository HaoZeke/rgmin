

Product manifold
----------------

The weighted Riemannian product :math:`\mathcal{M}_1 \times \cdots \times \mathcal{M}_k` with metric :math:`g = \sum_i w_i\, g_i`, the sphere of a subspace, tangent metrics on product factors, and the trust-region and session drivers that run on them.
:doc:`../explanation/product-manifold` derives every operation listed here.

Construction
~~~~~~~~~~~~

``rgmin::manifold::ProductFactor`` is one factor.
It is ``#[non_exhaustive]`` with public fields:

- ``kind: ManifoldKind``, the geometry;

- ``len: usize``, the packed length of a factor point;

- ``weight: f64``, the weight :math:`w_i` of the factor metric.

``ProductFactor::default()`` is a Euclidean factor of length 1 and weight 1.
``ProductFactor::new(kind, len)`` has weight 1, and ``with_kind``, ``with_len`` and ``with_weight`` set one field each.
``with_metric(Arc<dyn TangentMetric>)`` attaches a tangent metric and ``metric()`` returns it as ``Option<&Arc<dyn TangentMetric>>``; two factors compare equal only when they share the same metric object.

``ProductManifold::new(factors)`` builds the product of an iterator of ``ProductFactor`` in packing order.
A factor whose kind is itself a product is flattened into its blocks, and their weights are multiplied by the factor's weight; the blocks keep their tangent metrics.
It fails with ``Error::ProductFactor { index, what }``, where ``what`` is one of:

- ``"weight must be finite and positive"``;

- ``"packed length must be positive"``;

- ``"packed length does not fit the kind"``, from ``Manifold::required_dim`` of the kind;

- ``"a nested product takes tangent metrics on its factors"``, for a tangent metric on a factor whose kind is a product;

- ``"packed length overflows"``;

- ``"a product needs at least one factor"``, with index 0.

``ProductManifold::power(kind, block, count)`` is manopt's ``powermanifold``: ``count`` unit-weight copies of ``kind``, each of packed length ``block``.
``count = 0`` fails with ``"a power needs at least one copy"``.

``ManifoldKind::product(factors)`` and ``ManifoldKind::power(factor, block, count)`` wrap the two constructors in ``ManifoldKind::Product(Arc<ProductManifold>)``, whose ``as_str`` is ``"product"``.
``Oblique`` stays the dedicated kind for a power of spheres.

Accessors
~~~~~~~~~

.. table::

    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | Method                      | Returns                                                                                                            |
    +=============================+====================================================================================================================+
    | ``packed_len()``            | packed length of a point                                                                                           |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``factor_count()``          | number of factors after flattening                                                                                 |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``factors()``               | the flattened factors, with their weights in the metric and their tangent metrics                                  |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``tangent_metric_factor()`` | ``Some`` index of the first factor with a tangent metric                                                           |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``range(index)``            | ``Some`` slice of the packed vector holding factor ``index``, ``None`` past the end                                |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``is_flat()``               | every factor keeps a straight step on its set                                                                      |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``is_chart()``              | every factor has weight one in these coordinates, so ``chart`` is the identity                                     |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``chart_scales()``          | :math:`d` of :math:`z = d \odot x`: :math:`\sqrt{w_i}` on factor :math:`i` of a product from ``new``, 1 on a chart |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+
    | ``chart()``                 | the same manifold in the coordinates :math:`z = d \odot x`, all weights one                                        |
    +-----------------------------+--------------------------------------------------------------------------------------------------------------------+

Packing
~~~~~~~

A point packs the factor points end to end in the order given, each in its kind's own packing (:doc:`../howto/manifolds`).
A tangent vector and a Euclidean gradient pack the same way.
The hyperplane :math:`\mathbb{R}^n \times S^{n-1}` is ``[p_1, ..., p_n, n_1, ..., n_n]``:

.. code:: rust

    use rgmin::manifold::ProductFactor;
    use rgmin::ManifoldKind;

    let n = 4;
    let hyperplane = ManifoldKind::product([
        ProductFactor::new(ManifoldKind::Euclidean, n).with_weight(0.3),
        ProductFactor::new(ManifoldKind::Sphere, n),
    ])?;

Operations
~~~~~~~~~~

.. table::

    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``Manifold`` method          | On the product                                                                                             |
    +==============================+============================================================================================================+
    | ``required_dim``             | the packed length, exactly                                                                                 |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``project``                  | each factor's projection; no weights                                                                       |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``egrad2rgrad``              | each factor's ``egrad2rgrad`` of its block, then :math:`G^{-1}`, divided by :math:`w_i`                    |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``retract``                  | each factor's retraction                                                                                   |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``transport``                | each factor's transport                                                                                    |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``inner``                    | :math:`\sum_i w_i\, g_i(u_i, G v_i)`                                                                       |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``norm``                     | the square root of ``inner``                                                                               |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``ehess2rhess``              | each factor's ``ehess2rhess`` of its block of the full action, then :math:`G^{-1}`, divided by :math:`w_i` |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``exp``, ``log``             | each factor's                                                                                              |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``dist``                     | :math:`(\sum_i w_i\, d_i^2)^{1/2}`                                                                         |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``parallel_transport_along`` | each factor's, along :math:`t \mapsto \exp_x(t d)`                                                         |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``parallel_transport``       | each factor's, along the geodesic of ``log``                                                               |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``transport_is_isometric``   | true when it is for every factor and no factor has a tangent metric                                        |
    +------------------------------+------------------------------------------------------------------------------------------------------------+
    | ``typical_dist``             | :math:`(\sum_i w_i\, t_i^2)^{1/2}` over the factor typical distances :math:`t_i`                           |
    +------------------------------+------------------------------------------------------------------------------------------------------------+

:math:`G` is the factor's tangent metric, the identity when it has none.
The optional operations, ``ehess2rhess``, ``exp``, ``log``, ``dist`` and both parallel transports, return ``None`` as soon as one factor returns ``None``; all but ``ehess2rhess`` also return ``None`` when a factor has a tangent metric.
Arrays of the wrong length leave the input unchanged: ``project``, ``egrad2rgrad`` and ``transport`` return the vector, ``retract`` returns the point, ``inner`` and ``norm`` return NaN, and the optional operations return ``None``.

Factor capabilities
~~~~~~~~~~~~~~~~~~~

.. table::

    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | Kind                                                      | ``ehess2rhess`` | ``exp`` | ``log``, ``dist``, parallel transport | isometric ``transport`` | typical distance            |
    +===========================================================+=================+=========+=======================================+=========================+=============================+
    | ``Euclidean``                                             | yes             | yes     | yes                                   | yes                     | :math:`\sqrt{n}`            |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Sphere``, ``Stiefel``                                   | yes             | yes     | yes                                   | no                      | :math:`\pi`                 |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``SubspaceSphere``                                        | yes             | yes     | yes                                   | no                      | :math:`\pi`                 |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``StiefelP { n, p }``                                     | yes             | no      | no                                    | no                      | :math:`\sqrt{p}`            |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``So3``                                                   | yes             | yes     | yes                                   | no                      | :math:`\pi \sqrt{3}`        |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Se3``                                                   | yes             | yes     | yes                                   | no                      | :math:`(3 \pi^2 + 3)^{1/2}` |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Oblique { n, m }``                                      | yes             | yes     | yes                                   | no                      | :math:`\pi \sqrt{m}`        |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``ComplexCircle { n }``                                   | yes             | yes     | yes                                   | no                      | :math:`\pi \sqrt{n}`        |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``SphereComplex { n }``                                   | yes             | yes     | yes                                   | no                      | :math:`\pi`                 |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Positive { n }``                                        | yes             | yes     | yes                                   | no                      | :math:`\sqrt{n}`            |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Symmetric``, ``SkewSymmetric``                          | yes             | yes     | yes                                   | yes                     | :math:`\sqrt{\mathrm{len}}` |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``CenteredMatrix { m, n, rows }``                         | yes             | yes     | yes                                   | yes                     | :math:`\sqrt{\dim}`         |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``EuclideanComplex { n }``                                | yes             | yes     | yes                                   | yes                     | :math:`\sqrt{n}`            |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Constant { n }``                                        | yes             | yes     | yes                                   | yes                     | 0                           |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Spd``                                                   | yes             | no      | yes                                   | yes                     | :math:`\sqrt{\mathrm{len}}` |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``Multinomial``                                           | no              | no      | no                                    | no                      | :math:`\sqrt{\mathrm{len}}` |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``MultinomialDoublyStochastic``, ``MultinomialSymmetric`` | no              | no      | no                                    | no                      | :math:`\sqrt{\mathrm{len}}` |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+
    | ``MwRigid``, ``RigidQuotient``                            | no              | no      | no                                    | no                      | :math:`\sqrt{\mathrm{len}}` |
    +-----------------------------------------------------------+-----------------+---------+---------------------------------------+-------------------------+-----------------------------+

:math:`\mathrm{len}` is the packed length and :math:`\dim` the dimension of the centered matrices, :math:`m n - n` for centered rows and :math:`m n - m` for centered columns.
``Positive`` and the doubly stochastic kinds carry their own metric through ``inner`` and ``egrad2rgrad``; every other kind uses the dot product of its packed coordinates.
``Multinomial`` pairs a Fisher-orthogonal projection with the dot product, so its gradient does not represent the differential in its inner product.

Subspace sphere
~~~~~~~~~~~~~~~

``rgmin::manifold::SubspaceSphere`` is the unit sphere :math:`S(U) = \{x \in U : \|x\| = 1\}` of a linear subspace :math:`U \subset \mathbb{R}^n`, in the ambient Euclidean metric.

- ``SubspaceSphere::within(basis: Array2<f64>) -> Result<Self>`` takes :math:`U` as the span of the rows of ``basis``, each of length :math:`n`.

- ``SubspaceSphere::excluding(basis: Array2<f64>) -> Result<Self>`` takes :math:`U` as the orthogonal complement of that span; no row gives the whole sphere :math:`S^{n-1}`.

``ManifoldKind::sphere_within(basis)`` and ``ManifoldKind::sphere_excluding(basis)`` wrap them in ``ManifoldKind::SubspaceSphere(Arc<SubspaceSphere>)``, whose ``as_str`` is ``"subspace_sphere"``.
The rows are orthonormalized by modified Gram–Schmidt with a second pass.
Both constructors fail with ``Error::SubspaceBasis { what }``, where ``what`` is one of:

- ``"the ambient dimension must be positive"``;

- ``"a basis entry is not finite"``;

- ``"the basis rows are linearly dependent"``, when the part of a row outside the span of the earlier rows is at most :math:`\sqrt{\varepsilon}` of its length;

- ``"the subspace needs at least one direction"``, from ``within`` with no row;

- ``"the excluded directions leave no subspace"``, from ``excluding`` with :math:`n` independent rows.

.. table::

    +-------------------------+---------------------------------------------------------------------------------------------+
    | Method                  | Returns                                                                                     |
    +=========================+=============================================================================================+
    | ``ambient_dim()``       | :math:`n`, the packed length of a point                                                     |
    +-------------------------+---------------------------------------------------------------------------------------------+
    | ``subspace_dim()``      | :math:`\dim U`; the sphere has dimension one less                                           |
    +-------------------------+---------------------------------------------------------------------------------------------+
    | ``basis()``             | the orthonormal rows, spanning :math:`U`, or :math:`U^\perp` when ``excludes()``            |
    +-------------------------+---------------------------------------------------------------------------------------------+
    | ``excludes()``          | ``true`` for a sphere from ``excluding``                                                    |
    +-------------------------+---------------------------------------------------------------------------------------------+
    | ``project_subspace(v)`` | :math:`P_U v`; a vector of another length comes back unchanged                              |
    +-------------------------+---------------------------------------------------------------------------------------------+
    | ``closest_point(x)``    | :math:`P_U x / \Vert P_U x \Vert`; ``None`` for another length or no component in :math:`U` |
    +-------------------------+---------------------------------------------------------------------------------------------+

As a ``Manifold``, ``project`` is :math:`P_U - x x^\top`, ``retract`` is :math:`P_U(x + v) / \Vert P_U(x + v) \Vert`, ``transport`` is projection at the arrival point and ``ehess2rhess`` is :math:`(P_U - x x^\top)(\nabla^2 \bar f\, v - (x^\top \nabla \bar f)\, v)`.
The exponential, logarithm, distance and both parallel transports are the sphere's.

Tangent metrics
~~~~~~~~~~~~~~~

.. code:: rust

    pub trait TangentMetric: Send + Sync {
        fn apply(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
        fn solve(&self, x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64>;
    }

``rgmin::manifold::TangentMetric`` is an operator :math:`G_x` on the tangent spaces of one factor, attached with ``ProductFactor::with_metric``.
The factor's term of the product metric becomes :math:`w_i\, g_i(u, G_x v)`.
``apply`` returns :math:`G_x v` and ``solve`` returns :math:`G_x^{-1} r`, for the factor point ``x`` and tangent vectors in the factor's own packing.
At every ``x``, :math:`G_x` must map the tangent space into itself and be self-adjoint and positive definite in :math:`g_i`, and ``solve`` must invert ``apply``; the product projects both results onto the tangent space.
With a tangent metric the gradient is :math:`w_i^{-1} G_x^{-1} \mathrm{grad}_i f`, and ``ehess2rhess`` returns the model :math:`w_i^{-1} G_x^{-1} \mathrm{Hess}_i f`, which is self-adjoint in the metric and exact at critical points.
``chart`` keeps tangent metrics unchanged.

Sphere Hessians
~~~~~~~~~~~~~~~

``ehess2rhess`` of ``Sphere``, ``SubspaceSphere`` and the other embedded kinds that supply it keeps the Weingarten term explicit; on the sphere it is :math:`(I - x x^\top) \nabla^2 \bar f\, v - (x^\top \nabla \bar f)\, v`.
``egrad`` and ``ehess`` may come from any smooth extension :math:`\bar f` of the cost, and the result does not depend on the choice.
The session's dogleg, Newton-CG and Newton-PCG steps model the Hessian of a curved kind without the term, which is exact for the degree-zero extension :math:`\bar f(x) = f(x / \Vert x \Vert)` and converges linearly for others; ``minimize_rtr`` keeps the term.

Sessions
~~~~~~~~

``Solver::set_product(factors) -> Result<()>`` selects a weighted product; on ``Error::ProductFactor`` the session keeps its manifold.
``Solver::set_power(factor, block, count) -> bool`` selects a power, and returns ``false``, keeping the manifold, when ``factor`` cannot hold ``block`` or ``count`` is 0.
``Solver::set_manifold`` accepts a ``ManifoldKind::Product`` directly.
A change of manifold clears the session's history.
The session dimension must equal the packed length.

A product whose weights are not all one is stored as its chart.
The caller's points, gradients, Hessians and reports stay in natural coordinates, and the reported gradient norm is the length in the product metric.
A step on such a product fails with ``Error::ProductChart { what }`` while one of these settings is active:

- the exponential preconditioner;

- the Lindh model Hessian;

- rigid-motion projection of the whole vector (``set_project_rigid``);

- with the ``highs`` feature, the HiGHS box and equalities.

``MwRigid`` and ``RigidQuotient`` factors are projected block by block with the session's periodic flag.
``Solver::set_masses`` is read by the ``MwRigid`` blocks as one molecule reused on every block when its length is the atom count of one block, or as every block's masses in order when its length is the atom count of all ``MwRigid`` blocks; any other length means unit masses.

A step on a product with a tangent metric fails with ``Error::ProductTangentMetric { index }``, ``index`` being the first such factor after flattening; ``search_direction`` fails the same way and ``push_pair`` returns ``false``.

.. code:: rust

    pub fn run_gauged<O, G>(&mut self, obj: &O, x: &mut Array1<f64>, gauge: G) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
        G: FnMut(&mut Array1<f64>);

``Solver::run_gauged`` is ``run`` with ``gauge(x)`` called before every step but the first, to replace the iterate with another representation of the same point.
A call that changes ``x`` rebases the session as ``Solver::rebase``: the curvature memory stays, and the next step evaluates the objective afresh instead of reading the move as a step.
The returned report describes the point left in ``x``.

Trust region
~~~~~~~~~~~~

.. code:: rust

    pub fn minimize_rtr<M, FV, GG, HH>(
        manifold: &M,
        x0: Array1<f64>,
        value: FV,
        egrad: GG,
        ehess: HH,
        opts: RtrOptions,
    ) -> RtrMin
    where
        M: Manifold + ?Sized,
        FV: FnMut(&Array1<f64>) -> f64,
        GG: FnMut(&Array1<f64>) -> Array1<f64>,
        HH: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>;

``rgmin::rtr::minimize_rtr`` is the Absil-Baker-Gallivan trust region with truncated conjugate gradients, run in the metric of ``manifold``.
``egrad(x)`` is the Euclidean gradient and ``ehess(x, v)`` the Euclidean Hessian action, both in packed coordinates.
The Hessian is ``ehess2rhess``, or ``egrad2rgrad`` of the action when that returns ``None``.

``RtrOptions`` is ``#[non_exhaustive]``, and its defaults are those of manopt ``trustregions``:

.. table::

    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+
    | Field                  | Default  | Setter                      | Meaning                                                                                                           |
    +========================+==========+=============================+===================================================================================================================+
    | ``radius_max``         | ``None`` | ``with_radius_max``         | :math:`\bar\Delta`; ``None`` takes ``Manifold::typical_dist``, and :math:`\Delta_0 = \bar\Delta / 8`              |
    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``grad_tol``           | ``1e-6`` | ``with_grad_tol``           | stop when the Riemannian gradient is at most this long                                                            |
    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``max_iter``           | ``1000`` | ``with_max_iter``           | outer iterations                                                                                                  |
    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``cg_max``             | ``None`` | ``with_cg_max``             | inner iterations; ``None`` takes the packed length                                                                |
    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``theta``              | ``1.0``  | ``with_theta``              | :math:`\theta` of the inner rule :math:`\Vert r_j \Vert \le \Vert r_0 \Vert \min(\Vert r_0 \Vert^\theta, \kappa)` |
    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``kappa``              | ``0.1``  | ``with_kappa``              | :math:`\kappa` of the inner rule                                                                                  |
    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``rho_regularization`` | ``1e3``  | ``with_rho_regularization`` | both decreases gain :math:`\max(1, \lvert f \rvert)\, \varepsilon` times this                                     |
    +------------------------+----------+-----------------------------+-------------------------------------------------------------------------------------------------------------------+

A step is accepted when :math:`\rho > 0.1`; the radius is quartered below :math:`\rho = 1/4` and doubled, up to :math:`\bar\Delta`, above :math:`\rho = 3/4` on the boundary.
A non-finite or non-positive :math:`\bar\Delta` is replaced by 1.
``RtrMin`` is ``#[non_exhaustive]`` and holds ``x``, ``value``, ``grad_norm`` (in the manifold metric), ``iterations``, ``accepted`` and ``inner_iterations``, the conjugate-gradient iterations over all subproblems.

.. code:: rust

    pub fn minimize_rtr_gauged<M, FV, GG, HH, GA>(
        manifold: &M,
        x0: Array1<f64>,
        value: FV,
        egrad: GG,
        ehess: HH,
        gauge: GA,
        opts: RtrOptions,
    ) -> RtrMin
    where
        M: Manifold + ?Sized,
        FV: FnMut(&Array1<f64>) -> f64,
        GG: FnMut(&Array1<f64>) -> Array1<f64>,
        HH: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
        GA: FnMut(&mut Array1<f64>);

``rgmin::rtr::minimize_rtr_gauged`` is ``minimize_rtr`` with ``gauge(x)`` called after every accepted step, to replace the iterate with another representation of the same point.
The step is accepted on the value before the call.
The value is evaluated again when the call changed ``x``, and the gradient always is, so the next model, the stopping test and the returned ``RtrMin`` belong to the moved point.

.. code:: rust

    pub fn truncated_cg_riemannian<M, H>(
        manifold: &M,
        x: &Array1<f64>,
        egrad: &Array1<f64>,
        ehess: H,
        radius: f64,
        opts: &RtrOptions,
    ) -> TcgResult
    where
        M: Manifold + ?Sized,
        H: FnMut(&Array1<f64>) -> Array1<f64>;

``rgmin::rtr::truncated_cg_riemannian`` solves one subproblem of ``minimize_rtr`` at ``x`` with the trust radius ``radius``, using ``theta``, ``kappa`` and ``cg_max`` from ``opts``; on a product with a tangent metric it is the preconditioned subproblem.
``TcgResult`` holds the step ``eta``, the model decrease ``model_decrease``, the stopping reason ``stop`` and the conjugate-gradient ``iterations``.
A driver that measures the decrease itself retracts ``eta`` and passes the measured and model decreases with ``manifold.norm(x, &eta)`` to ``RtrRadius::update(actual_decrease, model_decrease, eta_norm) -> bool``, which applies the acceptance and radius rule above and returns whether the step is accepted.
``RtrRadius::new(radius_max)`` starts at a radius of ``radius_max / 8``.

C ABI
~~~~~

.. code:: c

    int32_t rgmin_solver_set_power(rgmin_solver_t *solver, int32_t factor, size_t block,
                                   size_t count);

ABI minor 41.
``factor`` is an ``rgmin_manifold_t`` value for a kind without shape parameters: ``RGMIN_MANIFOLD_EUCLIDEAN``, ``SPHERE``, ``SO3``, ``STIEFEL``, ``SE3``, ``RIGID_QUOTIENT``, ``MW_RIGID``, ``MULTINOMIAL``, ``SYMMETRIC`` or ``SKEWSYMMETRIC``.
It returns 0, or 1 with the manifold unchanged and ``rgmin_last_error`` set to:

- ``rgmin_solver_set_power: null solver``;

- ``rgmin_solver_set_power: the factor token needs a shape``, for a token with shape parameters or a value outside ``rgmin_manifold_t``;

- ``rgmin_solver_set_power: the block does not fit the factor``, for a block the factor cannot hold or a count of 0.

``MW_RIGID`` copies read ``rgmin_solver_set_masses`` as described under Sessions.
The C++ wrapper is ``rgmin::Solver::set_power(rgmin_manifold_t factor, std::size_t block, std::size_t count)``, and ``xts.h`` aliases it as ``xts_solver_set_power``.
Weighted products of arbitrary factors, subspace spheres, tangent metrics and gauges are not exposed to C or Python; :doc:`../explanation/product-manifold` gives the reasons.
