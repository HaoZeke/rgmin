===================================================================
Weighted product manifolds: metric, gradient, Hessian and geodesics
===================================================================


.. contents::

A point of a product manifold :math:`\mathcal{M} = \mathcal{M}_1 \times \cdots \times \mathcal{M}_k` is a tuple :math:`x = (x_1, \dots, x_k)` of factor points.
``ProductManifold`` (``src/manifold/product.rs``) packs the tuple end to end in one vector and gives it the metric :math:`g = \sum_i w_i\, g_i` with constant weights :math:`w_i > 0`.
This page derives every operation the solvers call from that one choice, shows where the weights enter and where they cancel, and compares the result with the products of manopt, Manifolds.jl, pymanopt and geomstats.
The first consumer is the optimal dividing hyperplane of transition-state theory in ``rgsaddle``, a point and a unit normal in :math:`\mathbb{R}^n \times S^{n-1}`.
Nothing below is specific to it.
:doc:`../reference/product-manifold` lists the API.

Each factor :math:`\mathcal{M}_i` sits in the space :math:`\mathcal{E}_i = \mathbb{R}^{n_i}` of its packed coordinates, with tangent projector :math:`P_i(x_i)` and metric :math:`g_i`.
For most kinds :math:`g_i` is the Euclidean inner product of :math:`\mathcal{E}_i` restricted to tangent vectors, which makes :math:`\mathcal{M}_i` a Riemannian submanifold of :math:`\mathcal{E}_i`.
``Positive`` and the doubly stochastic multinomials carry their own :math:`g_i`, through ``Manifold::inner`` and ``Manifold::egrad2rgrad``.
A caller supplies :math:`\bar f`, the objective on the packed coordinates, with its coordinate gradient :math:`\nabla \bar f = (\nabla_1 \bar f, \dots, \nabla_k \bar f)` and its Hessian action :math:`\nabla^2 \bar f\, u`.
The function optimized is the restriction :math:`f = \bar f|_{\mathcal{M}}`.

1 Why a product, and why weights
--------------------------------

A dividing hyperplane in mass-weighted coordinates :math:`y = M^{1/2} r` is :math:`\{y : n^\top (y - p) = 0\}`, named by a point :math:`p \in \mathbb{R}^n` and a unit normal :math:`n \in S^{n-1}`.
The pair :math:`(p, n)` lives on :math:`\mathbb{R}^n \times S^{n-1}`, a flat factor times a curved one.
Both factors are already kinds of ``ManifoldKind``; what has to be derived is their product.

The plain product metric :math:`|\delta p|^2 + |\delta n|^2` adds a squared mass-weighted length to a pure number.
Its value depends on the unit of length: going from ångström to bohr multiplies the first term by :math:`c^2`, with :math:`c \approx 1.89`, and leaves the second alone.
Every step length, trust radius, secant pair and stopping test measured with it inherits that dependence, so the path of the optimizer changes with the unit system.

Weights make that hidden parameter explicit.
Move the plane by :math:`(\delta p, \delta n)`, with :math:`n^\top \delta n = 0`.
A point :math:`y = p + r e` of the plane, :math:`e \perp n` a unit vector, is displaced along the normal by


.. math::

    (n + \delta n)^\top (y - p - \delta p) = r\, e^\top \delta n - n^\top \delta p + O(|\delta|^2).


The shift moves every point of the plane by at most :math:`|\delta p|`, and the turn moves points at distance :math:`r` by up to :math:`r |\delta n|`.
If the objective feels the plane out to a distance :math:`\ell`, in the units of :math:`p`, the two are comparable when :math:`|\delta p| \approx \ell |\delta n|`, and the natural size of a step is


.. math::

    \|(\delta p, \delta n)\|^2 = \ell^{-2} |\delta p|^2 + |\delta n|^2, \qquad w_p = \ell^{-2}, \quad w_n = 1.


Two scales are natural.
The geometric one is the mass-weighted radius of gyration of the system about :math:`p`.
The kinetic one uses that mass-weighted coordinates make the kinetic energy :math:`\tfrac12 |\dot y|^2`: an in-plane mode of angular frequency :math:`\omega` at temperature :math:`T` spreads over :math:`\ell = \sqrt{k_B T} / \omega`, again in the units of :math:`p`.
Either way :math:`\ell` scales with the unit of length, so :math:`w_p` carries the unit and the metric does not.
The section on the metric chart turns this into a statement about iterates.

The parameterization is redundant: :math:`p` can slide inside the plane, and :math:`(p, -n)` names the same plane.
An objective that depends on the plane alone is constant along the in-plane part of :math:`\delta p`, so its Hessian is singular there.
Trust-region and quasi-Newton steps tolerate that.
A caller that wants a minimal parameterization writes :math:`p = s\, n` and optimizes over :math:`\mathbb{R} \times S^{n-1}`.

2 The product metric
--------------------

For tangent vectors :math:`u = (u_1, \dots, u_k)` and :math:`v` at :math:`x`,


.. math::

    g_x(u, v) = \sum_{i=1}^k w_i\, g_{i, x_i}(u_i, v_i), \qquad w_i > 0.


The tangent space is the direct sum :math:`T_x \mathcal{M} = T_{x_1} \mathcal{M}_1 \oplus \cdots \oplus T_{x_k} \mathcal{M}_k` (Absil, Mahony and Sepulchre, section 3.1.6; Boumal, equation 3.31) :cite:`absilOptimizationAlgorithmsMatrix2008,boumalIntroductionOptimizationSmooth2023`, and :math:`g` makes the blocks orthogonal.
At :math:`w_i = 1` this is the product metric of Boumal, Example 3.57.
``ProductManifold::inner`` is the weighted sum and ``norm`` its square root.

The weighted product of the :math:`(\mathcal{M}_i, g_i)` is the plain product of the scaled factors :math:`(\mathcal{M}_i, w_i g_i)`.
Every fact about plain Riemannian products (Boumal, Table 7.2) therefore applies to it.
What remains is how the objects of one factor change when its metric is multiplied by a constant :math:`w > 0`:

.. table::

    +-----------------------------------------------------------+-------------------------------------------+
    | Object                                                    | Under :math:`g \mapsto w\, g`             |
    +===========================================================+===========================================+
    | Levi-Civita connection, geodesics, exponential, logarithm | unchanged                                 |
    +-----------------------------------------------------------+-------------------------------------------+
    | parallel transport                                        | unchanged                                 |
    +-----------------------------------------------------------+-------------------------------------------+
    | retraction, vector transport                              | unchanged; they do not involve the metric |
    +-----------------------------------------------------------+-------------------------------------------+
    | inner product                                             | times :math:`w`                           |
    +-----------------------------------------------------------+-------------------------------------------+
    | norm, distance, injectivity radius                        | times :math:`\sqrt{w}`                    |
    +-----------------------------------------------------------+-------------------------------------------+
    | gradient, Hessian operator, sectional curvature           | times :math:`1 / w`                       |
    +-----------------------------------------------------------+-------------------------------------------+

The Koszul formula,


.. math::

    2 g(\nabla_X Y, Z) = X g(Y, Z) + Y g(X, Z) - Z g(X, Y) + g([X, Y], Z) - g([X, Z], Y) - g([Y, Z], X),


is linear in :math:`g` on both sides, so a constant factor leaves the connection unchanged, and with it geodesics, the exponential, the logarithm and parallel transport.
Lengths of curves scale by :math:`\sqrt{w}`, and so do distances and the injectivity radius.
The gradient solves :math:`w\, g(\mathrm{grad}_w f, u) = \mathrm{D} f[u]`, so :math:`\mathrm{grad}_w f = w^{-1} \mathrm{grad} f`, and the Hessian :math:`\nabla_u \mathrm{grad}_w f` uses the same connection.
The curvature tensor depends on the connection alone, so the sectional curvature :math:`\langle R(u, v) v, u \rangle / (\|u\|^2 \|v\|^2 - \langle u, v \rangle^2)` gains :math:`w` above and :math:`w^2` below.
geomstats' ``ScalarProductMetric`` rescales the methods of a metric by exactly these rules.

3 The gradient
--------------

The Riemannian gradient is the tangent vector that represents the differential in the metric (Absil, Mahony and Sepulchre, section 3.6): :math:`g_x(\mathrm{grad} f(x), u) = \mathrm{D} f(x)[u]` for every :math:`u \in T_x \mathcal{M}`.
Write :math:`\mathrm{D} f(x)[u] = \sum_i \mathrm{D}_i f(x)[u_i]`, with :math:`\mathrm{D}_i f` the differential in the :math:`i`-th argument, and let :math:`\mathrm{grad}_i f(x) \in T_{x_i} \mathcal{M}_i` represent :math:`\mathrm{D}_i f(x)` in :math:`g_i`.
Then


.. math::

    \mathrm{grad} f(x) = \bigl( w_1^{-1} \mathrm{grad}_1 f(x), \dots, w_k^{-1} \mathrm{grad}_k f(x) \bigr),


because


.. math::

    g_x(\mathrm{grad} f(x), u) = \sum_i w_i\, g_i\bigl(w_i^{-1} \mathrm{grad}_i f(x), u_i\bigr) = \sum_i \mathrm{D}_i f(x)[u_i] = \mathrm{D} f(x)[u],


and the representer is unique because :math:`g_x` is nondegenerate.
At unit weights this is Boumal, Exercise 3.67.
A heavy factor gets a short gradient: its coordinates are expensive in the metric, so steepest descent moves them less.

4 Projection and egrad2rgrad
----------------------------

When every :math:`g_i` is induced by :math:`\mathcal{E}_i`, :math:`g` is induced by the weighted inner product


.. math::

    \langle a, b \rangle_W = \sum_i w_i\, a_i^\top b_i = a^\top W b, \qquad W = \mathrm{diag}(w_1 I_{n_1}, \dots, w_k I_{n_k}),


on :math:`\mathcal{E} = \mathcal{E}_1 \times \cdots \times \mathcal{E}_k`, and :math:`(\mathcal{M}, g)` is a Riemannian submanifold of the Euclidean space :math:`(\mathcal{E}, \langle \cdot, \cdot \rangle_W)`.
Two facts follow.

The :math:`W`-orthogonal projector onto :math:`T_x \mathcal{M}` is :math:`P(x) = \mathrm{diag}(P_1(x_1), \dots, P_k(x_k))`, free of weights.
The normal space of the product is the product of the factor normal spaces, and a vector normal to :math:`T_{x_i} \mathcal{M}_i` in :math:`\mathcal{E}_i` stays :math:`W`-normal because :math:`W` acts on block :math:`i` as the scalar :math:`w_i`.
``ProductManifold::project`` applies the factor projections block by block.

The :math:`W`-gradient of :math:`\bar f` is :math:`\nabla_W \bar f = W^{-1} \nabla \bar f`, since :math:`\langle W^{-1} \nabla \bar f, b \rangle_W = \nabla \bar f^\top b` for every :math:`b`.
On a Riemannian submanifold the Riemannian gradient is the projected ambient gradient (Absil, Mahony and Sepulchre, section 3.6), so


.. math::

    \mathrm{grad} f(x) = P(x)\, W^{-1} \nabla \bar f(x) = \bigl( w_i^{-1} P_i(x_i) \nabla_i \bar f(x) \bigr)_i,


the gradient of the previous section with :math:`\mathrm{grad}_i f = P_i \nabla_i \bar f`.
A factor with its own metric enters the same way through its own ``egrad2rgrad``, because the Riesz argument holds for any :math:`g_i`.
``ProductManifold::egrad2rgrad`` applies each factor's ``egrad2rgrad`` to its block of :math:`\nabla \bar f` and divides by :math:`w_i`.
Of the two maps a caller's gradient passes through, the projection and the gradient conversion, only the second sees the weights.

5 The product retraction
------------------------

A retraction is a smooth map :math:`R : T \mathcal{M} \to \mathcal{M}` with :math:`R_x(0_x) = x` and :math:`\mathrm{D} R_x(0_x) = \mathrm{id}_{T_x \mathcal{M}}` (Absil, Mahony and Sepulchre, Definition 4.1.1; Boumal, Definition 3.47).
The product retraction applies each factor's retraction to its block (Boumal, Exercise 3.50):


.. math::

    R_x(v) = \bigl( R^{(1)}_{x_1}(v_1), \dots, R^{(k)}_{x_k}(v_k) \bigr).

*First-order agreement.*
Each block is smooth in :math:`(x_i, v_i)`, so :math:`R` is smooth on :math:`T \mathcal{M} = T \mathcal{M}_1 \times \cdots \times T \mathcal{M}_k`, and :math:`R_x(0_x) = (R^{(i)}_{x_i}(0))_i = x`.
For :math:`v \in T_x \mathcal{M}` the curve :math:`c(t) = R_x(t v)` has the blocks :math:`c_i(t) = R^{(i)}_{x_i}(t v_i)`, so


.. math::

    c'(0) = \bigl( \mathrm{D} R^{(i)}_{x_i}(0)[v_i] \bigr)_i = (v_i)_i = v.


Hence :math:`\mathrm{D} R_x(0_x) = \mathrm{id}`, and in the packed coordinates :math:`R_x(t v) = x + t v + O(t^2)`: the retraction agrees with the step to first order.
The weights appear nowhere, since neither condition involves the metric.

A retraction is second order when its curves also start with zero covariant acceleration, :math:`\frac{\mathrm{D}}{\mathrm{d}t} c'(0) = 0` (Absil, Mahony and Sepulchre, equation 5.33; Boumal, Definition 5.42).
The connection of a product is the product of the factor connections (Boumal, Exercises 5.4 and 5.13), and the weights do not change it, so the acceleration of :math:`c` is the tuple of the factor accelerations.
The product retraction is therefore second order exactly when every factor retraction is.
The straight line of a flat factor and the normalized step on the sphere (Boumal, Example 5.43) are second order; the QR retraction of SO(3) is first order only.
The local rate of the trust region does not need a second-order retraction (Absil, Mahony and Sepulchre, Proposition 5.5.6), but the numerical Hessian check does (Boumal, section 6.8).
That is why the tests run the check along the exponential wherever a retraction is first order.

6 Exponential, logarithm, distance and injectivity radius
---------------------------------------------------------

Geodesics of the product are tuples of factor geodesics (Boumal, Exercise 5.39), and the weights leave them alone, so (Boumal, Exercise 10.32)


.. math::

    \exp_x(v) = \bigl( \exp_{x_i}(v_i) \bigr)_i, \qquad \log_x(y) = \bigl( \log_{x_i}(y_i) \bigr)_i.


If :math:`\exp_x(v) = y` then :math:`\exp_{x_i}(v_i) = y_i` for every :math:`i`, and :math:`\|v\|_x^2 = \sum_i w_i \|v_i\|_{x_i}^2` is smallest when every block is.
So, where each factor logarithm is defined, the tuple of factor logarithms is the logarithm of the product for every choice of positive weights.
The distance follows from Boumal, Exercise 10.14, applied to the scaled factors:


.. math::

    \mathrm{dist}(x, y) = \Bigl( \sum_i w_i\, \mathrm{dist}_i(x_i, y_i)^2 \Bigr)^{1/2}.

The exponential at :math:`x` is a diffeomorphism on the :math:`g`-ball of radius :math:`r` exactly when every factor exponential is one on its :math:`w_i g_i`-ball of radius :math:`r`.
A vector supported on one block shows that the condition is necessary.
It is sufficient because the :math:`g`-ball lies inside the product of the factor balls, where the exponential is a product of diffeomorphisms.
So


.. math::

    \mathrm{inj}(x) = \min_i \sqrt{w_i}\, \mathrm{inj}_i(x_i),


and a flat factor, with :math:`\mathrm{inj}_i = \infty`, does not constrain it.
A 2-plane tangent to factor :math:`i` has sectional curvature :math:`K_i / w_i`, and a plane spanned by vectors from two different factors has curvature zero.
For :math:`k \ge 2` the curvature bounds are :math:`\min(0, \min_i K_i^{\min} / w_i)` and :math:`\max(0, \max_i K_i^{\max} / w_i)`.
``Manifold`` has no injectivity or curvature hook; the two formulas are stated for step-size arguments and for the comparison below.

``ProductManifold::exp`` and ``log`` return ``None`` as soon as one factor has no closed form.
``dist`` is the weighted root sum of squares.

7 Vector transport
------------------

A vector transport with associated retraction :math:`R` maps :math:`\xi \in T_x \mathcal{M}` along :math:`\eta \in T_x \mathcal{M}` to :math:`\mathcal{T}_\eta \xi \in T_{R_x(\eta)} \mathcal{M}`, with :math:`\mathcal{T}_{0_x} \xi = \xi` and :math:`\mathcal{T}_\eta` linear (Absil, Mahony and Sepulchre, Definition 8.1.1).
The blockwise transport


.. math::

    \mathcal{T}_\eta \xi = \bigl( \mathcal{T}^{(i)}_{\eta_i} \xi_i \bigr)_i


satisfies the three conditions because each block does.
It lands in :math:`\bigoplus_i T_{R_{x_i}(\eta_i)} \mathcal{M}_i = T_{R_x(\eta)} \mathcal{M}`, it is the identity at :math:`\eta = 0`, and it is linear.
Its associated retraction is the product retraction, and it does not involve the metric.
For a Riemannian submanifold factor ``transport`` is projection onto the arrival tangent space (Absil, Mahony and Sepulchre, section 8.1.3), so on a product of them it is :math:`P(y)`, the :math:`W`-orthogonal projection.

The product transport is an isometry of :math:`g` exactly when every factor transport is an isometry of its :math:`g_i`.
If each is, then


.. math::

    g_y(\mathcal{T} \xi, \mathcal{T} \zeta) = \sum_i w_i\, g_i(\mathcal{T}^{(i)} \xi_i, \mathcal{T}^{(i)} \zeta_i) = \sum_i w_i\, g_i(\xi_i, \zeta_i) = g_x(\xi, \zeta).


If one is not, vectors supported on that block break the identity.
The weights can neither repair nor break isometry, so ``transport_is_isometric`` is the conjunction over the factors.
Projection is an isometry only where the tangent space does not turn: Euclidean space, the linear subspaces and the SPD cone.
On :math:`\mathbb{R}^n \times S^{n-1}` it is not.

8 Parallel transport and the locking condition
----------------------------------------------

Parallel transport along a curve :math:`c = (c_1, \dots, c_k)` is the tuple of factor parallel transports (Boumal, Exercise 10.39), unchanged by the weights because the connection is.
It is an isometry of :math:`g`: each block is an isometry of :math:`g_i`, and so of :math:`w_i g_i`.
``parallel_transport_along(x, d, v)`` carries :math:`v` along :math:`t \mapsto \exp_x(t d)` from :math:`t = 0` to :math:`t = 1`, and ``parallel_transport(x, y, v)`` takes :math:`d = \log_x(y)`.

The Riemannian Broyden class of Huang, Gallivan and Absil (`SIAM J. Optim. 25, 1660 (2015) <https://doi.org/10.1137/140955483>`_) pairs a retraction :math:`R` with an isometric transport :math:`\mathcal{T}_S` under the locking condition


.. math::

    \mathcal{T}_{S_\xi} \xi = \beta\, \mathcal{T}_{R_\xi} \xi, \qquad
    \beta = \frac{\|\xi\|}{\|\mathcal{T}_{R_\xi} \xi\|}, \qquad
    \mathcal{T}_{R_\xi} \xi = \frac{\mathrm{d}}{\mathrm{d}t} R_x(t \xi) \Big|_{t = 1}.


Take :math:`R = \exp` and :math:`\mathcal{T}_S` parallel transport along the geodesic.
The velocity of a geodesic is parallel along it, so :math:`\mathcal{T}_{R_\xi} \xi = P_{0 \to 1} \xi`, its length is :math:`\|\xi\|`, :math:`\beta = 1`, and the condition holds with equality.
On the product it holds block by block and so for the tuple, for every choice of weights.
A product whose factors all supply the exponential and parallel transport therefore carries a retraction and transport pair that meets the hypotheses of that theory.
The session's L-BFGS moves its pairs with ``transport``, projection, and does not rely on them.

9 The Riemannian Hessian and ehess2rhess
----------------------------------------

The Riemannian Hessian is :math:`\mathrm{Hess} f(x)[u] = \nabla_u \mathrm{grad} f` (Absil, Mahony and Sepulchre, Definition 5.5.1).
For a plain product, Boumal's Example 5.19 (equation 5.12, stated for two factors and extended to :math:`k` by grouping) gives, with :math:`G_i = \mathrm{grad}_i f`,


.. math::

    \mathrm{Hess} f(x)[u]_i = \mathrm{Hess}_i f(x)[u_i] + \sum_{j \ne i} \mathrm{D}_j G_i(x)[u_j].


The cross terms :math:`\mathrm{D}_j G_i` couple the blocks; only the diagonal terms carry the curvature of the factors.
On the scaled factors :math:`G_i` becomes :math:`w_i^{-1} G_i` and :math:`\mathrm{Hess}_i` becomes :math:`w_i^{-1} \mathrm{Hess}_i`, so for the weighted product


.. math::

    \mathrm{Hess} f(x)[u]_i = w_i^{-1} \Bigl( \mathrm{Hess}_i f(x)[u_i] + \sum_{j \ne i} \mathrm{D}_j G_i(x)[u_j] \Bigr).

For Riemannian submanifold factors the weighted ambient space gives the same result with the Weingarten terms explicit.
On a Riemannian submanifold of a Euclidean space (Boumal, Corollary 5.47; Absil, Mahony and Sepulchre, Proposition 5.3.2, for the connection it rests on),


.. math::

    \mathrm{Hess} f(x)[u] = P(x) \bigl( \nabla^2 \bar f(x)[u] \bigr) + \mathfrak{W}_x\bigl(u, P^\perp(x) \nabla \bar f(x)\bigr),


where the Weingarten map :math:`\mathfrak{W}_x` carries the curvature of the embedding.
In :math:`(\mathcal{E}, \langle \cdot, \cdot \rangle_W)` the ambient gradient is :math:`W^{-1} \nabla \bar f` and the ambient Hessian is :math:`W^{-1} \nabla^2 \bar f`.
The projector is blockwise and free of weights, and so is the Weingarten map: the normal spaces and projectors of block :math:`i` are those of :math:`\mathcal{M}_i \subset \mathcal{E}_i`, and :math:`W` acts on the block as a scalar, so the defining identity of :math:`\mathfrak{W}_i` holds with both sides multiplied by :math:`w_i`.
Block :math:`i` of the Hessian is therefore


.. math::

    \mathrm{Hess} f(x)[u]_i
    = w_i^{-1} \Bigl( P_i(x_i) \bigl( \nabla^2 \bar f(x)\, u \bigr)_i + \mathfrak{W}_i\bigl(u_i, P_i^\perp(x_i) \nabla_i \bar f(x)\bigr) \Bigr)
    = w_i^{-1}\, \mathrm{ehess2rhess}_i\bigl(x_i, \nabla_i \bar f, (\nabla^2 \bar f\, u)_i, u_i\bigr).


The block :math:`(\nabla^2 \bar f\, u)_i = \sum_j \nabla^2_{ij} \bar f\, u_j` of the caller's full Hessian action carries the cross terms; the factor adds only its own Weingarten term.
The same holds for every kind in the crate that supplies ``ehess2rhess``, including those with their own metric.
Each such map is affine in the Hessian action with linear part the kind's ``egrad2rgrad``, and :math:`\mathrm{D}_j G_i(x)[u_j] = \mathrm{egrad2rgrad}_i(\nabla^2_{ij} \bar f\, u_j)`, so the off-diagonal blocks of the action produce exactly the cross terms of equation 5.12.
``ProductManifold::ehess2rhess`` applies each factor's map to its block of the full action and divides by :math:`w_i`.

For the hyperplane, with the sphere's Weingarten term :math:`\mathfrak{W}_n(u_n, \nu) = -(n^\top \nu)\, u_n`,


.. math::

    \mathrm{Hess} f(p, n)[u]
    = \Bigl( w_p^{-1} \bigl( \nabla^2 \bar f\, u \bigr)_p,\;
    w_n^{-1} \bigl( (I - n n^\top) (\nabla^2 \bar f\, u)_n - (n^\top \nabla_n \bar f)\, u_n \bigr) \Bigr).

The weights cancel in the bilinear form,


.. math::

    g_x\bigl(\mathrm{Hess} f(x)[u], v\bigr) = v^\top \nabla^2 \bar f(x)\, u + \sum_i \mathfrak{W}_i\bigl(u_i, P_i^\perp \nabla_i \bar f(x)\bigr)^\top v_i,


which is symmetric in :math:`u` and :math:`v` because each Weingarten map is self-adjoint.
They cancel in the Newton equation too: :math:`\mathrm{Hess} f(x)[\eta] = -\mathrm{grad} f(x)` has :math:`w_i^{-1}` on both sides of block :math:`i`, so the Newton step does not depend on the weights.
Only the steps that measure, the gradient step and the trust region, see them.

10 Typical distance and the trust-region defaults
-------------------------------------------------

manopt's ``typicaldist`` is the distance scale a trust region starts from.
If every factor moves by its own typical distance :math:`t_i`, the product moves by


.. math::

    t(\mathcal{M}) = \Bigl( \sum_i w_i\, t_i^2 \Bigr)^{1/2},


the weighted form of manopt's root sum of squares and the value of ``ProductManifold::typical_dist``.
The factor values follow the manopt factories: :math:`\pi` for a sphere, :math:`\pi \sqrt{3}` for SO(3), :math:`\pi \sqrt{m}` for :math:`m` oblique columns and :math:`\sqrt{n}` for :math:`\mathbb{R}^n`.

``rtr::minimize_rtr`` is the trust region with truncated conjugate gradients of Absil, Baker and Gallivan (`Found. Comput. Math. 7, 303 (2007) <https://doi.org/10.1007/s10208-005-0179-9>`_; Absil, Mahony and Sepulchre, Algorithms 10 and 11).
It runs in the metric of the manifold: the gradient by ``egrad2rgrad``, the Hessian by ``ehess2rhess``, and every inner product, norm and the trust ball by ``inner``.
``RtrOptions`` takes the defaults of manopt ``trustregions``:

- :math:`\bar\Delta` the typical distance and :math:`\Delta_0 = \bar\Delta / 8`;

- acceptance above :math:`\rho' = 0.1`, the radius quartered below :math:`\rho = 1/4` and doubled, up to :math:`\bar\Delta`, above :math:`\rho = 3/4` on the boundary;

- the inner rule :math:`\|r_j\| \le \|r_0\| \min(\|r_0\|^\theta, \kappa)` with :math:`\theta = 1` and :math:`\kappa = 0.1` (Absil, Mahony and Sepulchre, equation 7.10), and an inner budget of the packed length, which bounds the dimension;

- ``rho_regularization`` :math:`10^3`, which adds :math:`10^3 \varepsilon \max(1, |f|)` to both decreases so that steps below the rounding of :math:`f` read :math:`\rho \approx 1`.

With the exact Hessian the local rate is :math:`\min(\theta + 1, 2)` (Absil, Mahony and Sepulchre, Theorem 7.4.11), quadratic at :math:`\theta = 1`.
A factor without ``ehess2rhess`` makes the product fall back to ``egrad2rgrad`` of the ambient action, the Gauss-Newton part of the Hessian, which loses that rate.

The typical distance of a flat factor is the one place a unit enters again.
:math:`\pi` is intrinsic to a sphere, but :math:`\sqrt{n}` for :math:`\mathbb{R}^n` counts coordinates in whatever unit they are written in.
Under :math:`p' = c\, p` and :math:`w_p' = w_p / c^2` the term :math:`w_p\, n` becomes :math:`w_p\, n / c^2`, and the default :math:`\bar\Delta` moves.
A run that must not depend on the unit sets :math:`\bar\Delta` from a physical length with ``RtrOptions::with_radius_max``, for instance :math:`\bar\Delta = (w_p L^2 + w_n \pi^2)^{1/2}` with :math:`L` the largest shift of :math:`p` worth trusting.

11 The metric chart and unit invariance
---------------------------------------

The session steppers, L-BFGS, conjugate gradients, FIRE, the Newton-CG trust region and the rest, measure steps, curvature pairs and gradients with the dot product of the packed vector.
On a weighted product that is not :math:`g`.
Rather than thread :math:`g` through every stepper, a session stores a weighted product in the coordinates where :math:`g` is the dot product.

The map :math:`\phi(x) = (\sqrt{w_1}\, x_1, \dots, \sqrt{w_k}\, x_k)` is a diffeomorphism from :math:`\mathcal{M}` onto the product of the scaled sets :math:`\sqrt{w_i}\, \mathcal{M}_i`, with :math:`\mathrm{D} \phi(x)[u] = (\sqrt{w_i}\, u_i)_i`.
For Riemannian submanifold factors


.. math::

    \mathrm{D} \phi(x)[u]^\top \mathrm{D} \phi(x)[v] = \sum_i w_i\, u_i^\top v_i = g_x(u, v),


so :math:`\phi` is an isometry onto :math:`\prod_i \sqrt{w_i}\, \mathcal{M}_i` with the Euclidean metric of the packed vector, and a stepper that measures with dot products in :math:`z = \phi(x)` measures in :math:`g`.
``ProductManifold::chart`` is that product and ``ProductManifold::chart_scales`` is :math:`d`, equal to :math:`\sqrt{w_i}` on the coordinates of factor :math:`i`.
A block stores :math:`s_i x_i` for a factor point :math:`x_i` and carries the metric :math:`c_i\, g_i`, with :math:`c_i s_i^2 = w_i`: :math:`s_i = 1` and :math:`c_i = w_i` in natural coordinates, :math:`s_i = \sqrt{w_i}` and :math:`c_i = 1` in the chart.
Every operation divides the block by :math:`s_i`, calls the factor and scales back.
The Weingarten term of the scaled factor :math:`s_i \mathcal{M}_i` is :math:`1 / s_i` times the unit one, which the code obtains by passing the factor :math:`\nabla_i \bar f / s_i`.

``Solver::set_manifold`` and ``Solver::set_product`` store a product whose weights are not all one as its chart.
The caller keeps natural coordinates: the session reads the objective as :math:`\tilde f(z) = \bar f(z \oslash d)`, so that


.. math::

    \nabla_z \tilde f = \nabla_x \bar f \oslash d, \qquad
    \nabla_z^2 \tilde f = D^{-1} \nabla_x^2 \bar f\, D^{-1}, \qquad
    \nabla_z^2 \tilde f\, v = D^{-1} \nabla_x^2 \bar f\, (D^{-1} v),


with :math:`D = \mathrm{diag}(d)`.
Box bounds are scaled by :math:`d`, and the point, the reported coordinates and the reported gradient are mapped back.
The reported gradient norm is already its :math:`g`-length.
Factors with their own metric (``Positive``, the doubly stochastic multinomials) are stepped in the chart as a session steps them alone, with their Riemannian gradient and dot products.
``minimize_rtr`` needs no chart, since it measures everything with ``inner``.

Settings that read raw coordinates as atoms or boxes cannot be rescaled behind the caller's back, and a weighted product refuses them with ``Error::ProductChart``.
They are the exponential preconditioner, the Lindh model Hessian, rigid-motion projection of the whole vector, and the HiGHS box and equalities.
Rigid factors (``MwRigid``, ``RigidQuotient``) inside a product are projected block by block with the session's masses and periodic flag.
``MwRigid`` blocks read the masses as one molecule reused on every block, or as the concatenation over the ``MwRigid`` blocks.

*Unit invariance.*
Change the unit of a factor, :math:`x_i' = c\, x_i`, and its weight with it, :math:`w_i' = w_i / c^2`.
The chart coordinate does not change, :math:`\sqrt{w_i'}\, x_i' = \sqrt{w_i}\, x_i`, and the objective written in the new unit, :math:`\bar f'(x') = \bar f(x)`, reads in the chart as the same function :math:`\tilde f`.
A deterministic stepper started at the same chart point therefore produces the same chart iterates, and mapping back gives :math:`x_i' = c\, x_i` at every step.
The iterates do not depend on the unit once the weight carries it; in floating point they agree to rounding.
The test ``iterates_do_not_depend_on_the_length_unit_once_the_weight_carries_it`` runs fifteen L-BFGS steps on the hyperplane in ångström with :math:`w_p` and in bohr with :math:`w_p / c^2`, and finds them equal to :math:`10^{-9}`.
Keeping :math:`w_p` in bohr moves them by more than :math:`10^{-4}`.
The same test runs ``minimize_rtr`` in both units with :math:`\bar\Delta` set explicitly, and finds the same iteration count, accepted steps and minimizer.

12 Comparison with manopt, Manifolds.jl, pymanopt and geomstats
---------------------------------------------------------------

At unit weights the product here is the product of:

- manopt ``productmanifold`` (`Boumal et al., JMLR 15, 1455 (2014) <https://jmlr.org/papers/v15/boumal14a.html>`_), source `a149c85 <https://github.com/NicolasBoumal/manopt/blob/a149c85ca3cf23ca828d8bc6a6b125a4ac82d656/manopt/tools/productmanifold.m>`_;

- ManifoldsBase.jl ``ProductManifold``, used by Manifolds.jl and by the solvers of Manopt.jl (`Axen et al., ACM TOMS 49, 1 (2023) <https://doi.org/10.1145/3618296>`_; `Bergmann, JOSS 7, 3866 (2022) <https://doi.org/10.21105/joss.03866>`_), source `b16e9d6 <https://github.com/JuliaManifolds/ManifoldsBase.jl/blob/b16e9d6f305a68385135092055cb8698d6c69e89/src/ProductManifold.jl>`_;

- pymanopt ``Product`` (`Townsend et al., JMLR 17, 1 (2016) <https://jmlr.org/papers/v17/16-177.html>`_), source `35b022f <https://github.com/pymanopt/pymanopt/blob/35b022ff6043b893f32c903b428e811dea200dc2/src/pymanopt/manifolds/product.py>`_;

- geomstats ``ProductRiemannianMetric`` (`Miolane et al., JMLR 21, 1 (2020) <https://jmlr.org/papers/v21/19-027.html>`_), source `5dd5cc2 <https://github.com/geomstats/geomstats/blob/5dd5cc286a86f71ce9aed4f263539ec4bfa4c586/geomstats/geometry/product_manifold.py>`_.

All five agree on the summed inner product and on the blockwise exponential and logarithm, and those that offer parallel transport carry it per factor.
manopt and pymanopt also convert Euclidean gradients and Hessians factor by factor, passing each factor its block of the full Euclidean Hessian action as here, so the cross terms of equation 5.12 arrive the same way.
ManifoldsBase.jl supplies the Weingarten map per factor for the same purpose, and geomstats has no such conversion.

.. table::

    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | Aspect             | rgmin                       | manopt                  | Manifolds.jl                          | pymanopt                | geomstats                           |
    +====================+=============================+=========================+=======================================+=========================+=====================================+
    | weights            | :math:`w_i > 0` per factor  | none                    | none                                  | none                    | per factor, ``ScalarProductMetric`` |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | points             | one packed vector           | struct of fields        | ``ArrayPartition``                    | list                    | one vector, or stacked              |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | nested product     | flattened, weights multiply | nested structs          | ``×`` flattens                        | refused                 | an ordinary factor                  |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | missing operation  | ``None`` at the call        | field absent            | method error                          | factor raises           | numerical solver, else error        |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | distance           | weighted :math:`\ell_2`     | :math:`\ell_2`          | :math:`\ell_r`, default :math:`r = 2` | :math:`\ell_2`          | :math:`\ell_2`                      |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | typical distance   | weighted :math:`\ell_2`     | :math:`\ell_2`          | none                                  | :math:`\ell_2`          | none                                |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | random tangent     | none                        | scaled :math:`k^{-1/2}` | per factor                            | scaled :math:`k^{-1/2}` | per factor                          |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | retraction choice  | one per kind                | one per factory         | ``ProductRetraction``                 | one per factor          | exponential                         |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | parallel transport | per factor, optional        | if all factors have it  | per factor                            | none                    | per factor                          |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | injectivity radius | not exposed                 | not exposed             | minimum                               | not exposed             | minimum                             |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+
    | power manifold     | unit-weight product         | ``powermanifold``       | ``PowerManifold``                     | none                    | ``NFoldManifold``                   |
    +--------------------+-----------------------------+-------------------------+---------------------------------------+-------------------------+-------------------------------------+

The differences in detail:

- *Weights.*
  Only this product weighs its factors.
  manopt, Manifolds.jl and pymanopt build the element-wise sum of the factor metrics, and a caller who wants :math:`w_i g_i` must write a scaled factor.
  geomstats removed the ``scales`` of ``ProductManifold`` and ``ProductRiemannianMetric`` in 2022 (`856c3ac <https://github.com/geomstats/geomstats/commit/856c3ac2104e>`_).
  It now scales a factor by equipping it with ``ScalarProductMetric``, whose rules are the table of the metric section: inner products times :math:`s`; norms, distances and injectivity radius times :math:`\sqrt{s}`; sectional curvature times :math:`1/s`; exponential, logarithm and parallel transport unchanged.
  Its product reshape for such factors was fixed in November 2026 (`pull request 2203 <https://github.com/geomstats/geomstats/pull/2203>`_).
  A weighted product here is a geomstats product of ``ScalarProductMetric`` factors.

- *Points.*
  A point here is one ``f64`` vector holding each factor point in its kind's packing, column-major for ``StiefelP`` like manopt's ``X(:)`` and row-major for the symmetric kinds, so the session steppers and the C ABI see one array.
  manopt uses a struct with one named field per factor, Manifolds.jl a RecursiveArrayTools ``ArrayPartition``, and pymanopt a list.
  geomstats concatenates the flattened factor points, or stacks them when every factor has the same shape.

- *Nesting.*
  ``ProductManifold::new`` flattens a product factor into its blocks and multiplies the weights, so :math:`(\mathcal{M}_1 \times \mathcal{M}_2) \times \mathcal{M}_3` is a three-factor product.
  Manifolds.jl's ``×`` also flattens, without weights.
  manopt nests structs, pymanopt refuses a nested product, and geomstats treats it as an ordinary factor.

- *Optional operations.*
  Here every ``Manifold`` method exists on every product.
  The exponential, logarithm, distance, parallel transport and ``ehess2rhess`` return ``None`` as soon as one factor lacks a closed form, and ``minimize_rtr`` then uses the Gauss-Newton Hessian.
  manopt defines a field such as ``M.exp`` or ``M.ehess2rhess`` only when all elements provide it, so a solver sees the absence before it starts.
  pymanopt always defines the method and the factor raises ``NotImplementedError`` at the call.
  Manifolds.jl fails with a method error, and geomstats falls back on a factor metric's numerical ``exp_solver`` and ``log_solver`` when it has them.

- *Gradient and Hessian conversion.*
  manopt's and pymanopt's conversions have the blockwise structure of this one without the division by :math:`w_i`.
  manopt warns that a factor's ``egrad2rgrad`` or ``ehess2rhess`` written with ``storedb`` and ``key`` arguments will not receive them inside a product.

- *Distance and norm.*
  Here the distance is the weighted root sum of squares, which is the Riemannian distance of :math:`g`.
  manopt, pymanopt and geomstats take the unweighted root sum of squares.
  Manifolds.jl's ``distance(M, p, q, r)`` and ``norm(M, p, X, r)`` take any :math:`\ell_r` norm of the factor values, :math:`r = 2` by default; for :math:`r \ne 2` they are not the distance and norm of the product metric.

- *Typical distance and default radius.*
  Here it is the weighted root sum of squares; manopt and pymanopt take the unweighted one, and both start their trust regions at :math:`\bar\Delta` equal to it and :math:`\Delta_0 = \bar\Delta / 8`, pymanopt falling back on :math:`\sqrt{\dim \mathcal{M}}`.
  Manopt.jl's ``trust_regions`` defaults to :math:`\bar\Delta = \sqrt{\dim \mathcal{M}}` (`01ae0f6 <https://github.com/JuliaManifolds/Manopt.jl/blob/01ae0f62e90c9def032e8e32b3e4ffc988ca9288/src/solvers/trust_regions.jl>`_), and ManifoldsBase.jl and geomstats have no typical distance.

- *Random tangent vectors.*
  manopt and pymanopt scale each factor's random unit tangent vector by :math:`k^{-1/2}`, so the tuple has unit length in the unweighted metric.
  There is no random point or tangent here.

- *Retraction and transport choice.*
  Here each kind has one retraction and one transport.
  manopt uses each factory's ``retr`` and ``transp``, with ``retr2``, ``isotransp`` and ``paralleltransp`` when all elements provide them.
  Manifolds.jl composes ``ProductRetraction``, ``InverseProductRetraction`` and ``ProductVectorTransport`` from any per-factor methods.
  pymanopt uses each factor's ``retraction`` and ``transport``, and geomstats works with the exponential and logarithm.

- *Parallel transport and isometry.*
  Here parallel transport is per factor, ``None`` when a factor has none, and ``transport_is_isometric`` is the conjunction over the factors.
  manopt offers ``paralleltransp`` and ``isotransp`` when every element does, Manifolds.jl and geomstats transport per factor, and pymanopt has no parallel transport.

- *Injectivity radius and curvature.*
  Neither is exposed here.
  Manifolds.jl takes the minimum injectivity radius of the factors, computes the sectional curvature of a product exactly, bounds it by the factor bounds and 0 when :math:`k \ge 2`, and calls a product flat when every factor is.
  geomstats takes the minimum injectivity radius, and ``ScalarProductMetric`` scales it by :math:`\sqrt{s}`.
  ``ProductManifold::is_flat`` here means that every factor keeps a straight step on its set, a different notion from the zero curvature of Manifolds.jl's ``is_flat``.

- *Power manifold.*
  ``ProductManifold::power`` and ``ManifoldKind::power`` are unit-weight products, and Oblique stays a dedicated kind with its own column loops.
  manopt's ``powermanifold``, Manifolds.jl's ``PowerManifold`` and geomstats' ``NFoldManifold`` are separate types, and pymanopt has none.

- *Metric matrix and signature.*
  geomstats' ``ProductRiemannianMetric`` also offers a block-diagonal ``metric_matrix`` and the summed signature.
  The trait here has only ``inner`` on tangent vectors.

13 Reconciliation with the unlanded manifold branches
-----------------------------------------------------

Two earlier branches carried product designs.
Neither landed, and this one supersedes both.

``feat/manifold-api-060`` (``cdc8f5dd``) added ``RiemannianManifold``, a product of Stiefel and SPD factors over caller-owned column-major slices.
It had QR and polar Stiefel retractions, the Edelman, Arias and Smith exponential, SPD under the affine-invariant and log-Euclidean metrics, and a ``gradient_step`` helper.
It sat beside ``ManifoldKind``, so no session could step it, and it implemented the old ``Manifold`` trait, whose only metric was the dot product.
With the affine-invariant SPD gradient :math:`X \nabla f X`, a stepper's secant pairs would have mixed two metrics.
Here the product is a ``ManifoldKind`` over every kind, with weights, and the trait carries ``inner``, so gradient and metric agree.
Stiefel frames keep the column-major packing that both designs use.
The polar retraction and the two SPD metrics are not ported: the ``Spd`` kind stays Frobenius, whose gradient is the symmetrized Euclidean one.
An affine-invariant SPD kind can now be added through ``inner`` and ``egrad2rgrad``, and it would enter a product unchanged.

``feat/power-manifold`` (``3bb6f1df``) added ``ManifoldKind::Power`` over four factors (Euclidean, sphere, MwRigid, rigid quotient), ``Solver::set_power``, a C setter with private factor codes 0 to 3 and the token ``RGMIN_MANIFOLD_POWER``, and ``minimize_rtr`` in the dot-product metric.
The power is now a special case of the product.
``ManifoldKind::power`` builds a unit-weight ``ProductManifold`` of any kind, so a power inherits the metric, exponential, logarithm, Hessian and chart instead of repeating them.
``Solver::set_power`` keeps its signature, its rule for ``MwRigid`` masses (one molecule reused on every copy, or every copy's masses in order) and its per-block Eckart projection.
``rgmin_solver_set_power`` keeps its signature but reads the factor as an ``rgmin_manifold_t`` token for any kind without shape parameters, in place of the private codes.
The branch never shipped, so no ABI user sees the change.
``RGMIN_MANIFOLD_POWER`` is not added: a token cannot carry the factor, block and count, and the branch's default, two Euclidean blocks of length one, is just :math:`\mathbb{R}^2`.
``minimize_rtr`` is ported and now runs in ``inner`` with ``ehess2rhess`` and the typical distance.
Oblique stays its own kind, the power of spheres with column loops of its own.

14 How the derivation is checked
--------------------------------

``tests/product_manifold.rs`` checks each section numerically on a quartic :math:`\bar f(x) = \tfrac12 x^\top A x + b^\top x + \tfrac14 \sum_i x_i^4` whose dense :math:`A` couples every block:

- manopt's ``checkgradient`` and ``checkhessian``: the remainders of the first- and second-order models along :math:`t \mapsto R_x(t v)` and :math:`t \mapsto \exp_x(t v)` have slopes 2 and 3 in :math:`\log t` (Boumal, sections 4.8 and 6.8), on the weighted :math:`\mathbb{R}^4 \times S^3` along both curves, on :math:`S^2 \times SO(3)` with the Hessian along the exponential, and on every kind with a closed-form geodesic times a weighted :math:`\mathbb{R}^2`;

- :math:`g(\mathrm{grad} f, v) = \nabla \bar f^\top v`, tangency of the gradient and of the Hessian action, and symmetry of the Hessian in :math:`g`;

- the retraction and the exponential stay on the manifold for steps from :math:`10^{-3}` to :math:`10`;

- transport lands in the target tangent space, parallel transport preserves :math:`g`, and a product of flat factors with weights 2, 1/4 and 1 transports isometrically;

- the locking condition with :math:`\beta = 1`: the velocity of :math:`t \mapsto \exp_x(t d)` at :math:`t = 1`, by central differences, equals ``parallel_transport_along(x, d, d)``;

- :math:`\log_x(\exp_x(v)) = v`, :math:`\mathrm{dist}(x, \exp_x(v)) = \|v\|` and the weighted typical distance;

- L-BFGS, steepest descent, the session Newton-CG trust region and ``minimize_rtr`` reach the minimum of a coupled Rayleigh quotient and quadratic on the hyperplane, :math:`\tfrac12 (p - c)^\top B (p - c) + n^\top A n + \kappa ((p - c)^\top n)^2`, whose value is the lowest eigenvalue of :math:`A`;

- the unit test of the chart section, and the rigid factors keep their mass-weighted centres.

15 Limitations
--------------

- The session's Newton-CG trust region models the Hessian by the projected ambient action, without the Weingarten term.
  On a curved factor with a nonzero multiplier it converges linearly.
  On the hyperplane test, on the tangent space of the sphere at the minimum, its model is :math:`2A` where the Hessian is :math:`2(A - \lambda_1 I)`, so each step multiplies the error by at most :math:`\lambda_1 / \lambda_2 = 1/2`, the ratio of the two lowest eigenvalues of :math:`A`, which is the rate it shows.
  ``minimize_rtr`` uses ``ehess2rhess`` and converges quadratically on the same problem: its gradient norm goes from :math:`2 \times 10^{-4}` to :math:`2 \times 10^{-9}` to :math:`10^{-16}` in its last three iterations.

- The QR retraction of SO(3) is first order, so the numerical Hessian check on :math:`S^2 \times SO(3)` runs along the exponential.

- ``Multinomial`` projects Fisher-orthogonally but keeps the dot product as its inner product, so its gradient does not represent the differential in that inner product, and the derivative tests leave it out.
  The doubly stochastic kinds carry the Fisher inner product and are consistent.

- ``Spd`` is the Frobenius metric on the cone.
  Its retraction :math:`x + v + \tfrac12 v x^{-1} v` is not second order for that metric, and it has no exponential, because straight lines leave the cone.
  Its logarithm :math:`y - x`, its parallel transport, the identity, and its Hessian are exact.

- ``Multinomial`` and its doubly stochastic variants, ``MwRigid`` and ``RigidQuotient`` have no Hessian or geodesic closed forms, and ``StiefelP`` has a Hessian but no exponential, logarithm or parallel transport.
  A product containing one returns ``None`` for those operations; projection, retraction, transport and the gradient still work.

- A weighted product refuses the settings that read raw coordinates, with ``Error::ProductChart``.

- The C ABI selects powers only, through ``rgmin_solver_set_power``.
  A general weighted product needs a descriptor for parameterized factors (the sizes of Oblique, Stiefel and the complex kinds) and their weights, and that struct is a separate ABI design.
  The Python extension selects no manifold at all yet; a product waits for that API.
