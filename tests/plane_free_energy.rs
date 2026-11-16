//! The harmonic free energy of a dividing plane near a saddle, the
//! optimal-hyperplane TST model. A plane is a unit normal `n` on the
//! sphere of the subspace the sampler moves in, orthogonal to the
//! centre-of-mass direction, and a signed offset `d` from a pivot `c`:
//! \(\{q : n\cdot(q - c) = d\}\). With \(s = n^\top H^+ n\) and
//! \(r = d - n\cdot(q^\ddagger - c)\), the free energy up to constants is
//! \(A = r^2/(2s) + \tfrac{kT}{2}\ln(-s)\), the drivers minimise
//! \(f = -A\), and the optimum is the unstable mode through the saddle.

use std::sync::{Arc, Mutex};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, s};
use rgmin::manifold::{ProductFactor, ProductManifold, TangentMetric};
use rgmin::rtr::{
    RtrMin, RtrOptions, RtrRadius, minimize_rtr, minimize_rtr_gauged, truncated_cg_riemannian,
};
use rgmin::{Accept, Control, Error, Manifold, ManifoldKind, Method, Solver};

const D: usize = 24;
const KT: f64 = 0.3;
const UNSTABLE: f64 = -1.5;

/// Saddle eigenvalue of mode `j`: the unstable mode, then the `D - 2`
/// in-plane modes from 0.05 to 8, evenly in the logarithm. No single
/// weight on the rotation block matches all of their curvatures.
fn eigenvalue(j: usize) -> f64 {
    if j == 0 {
        UNSTABLE
    } else {
        0.05 * 160f64.powf((j - 1) as f64 / (D - 3) as f64)
    }
}

/// SplitMix64, uniform on [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn vector(&mut self, n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |_| self.next())
    }
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0, |m, e| m.max(e.abs()))
}

fn reflection(w: &Array1<f64>) -> Array2<f64> {
    let ww = w.dot(w);
    Array2::from_shape_fn((w.len(), w.len()), |(i, j)| {
        let delta = if i == j { 1.0 } else { 0.0 };
        delta - 2.0 * w[i] * w[j] / ww
    })
}

/// Centre-of-mass direction of `D` unit masses on a line.
fn centre_of_mass() -> Array1<f64> {
    Array1::from_elem(D, 1.0 / (D as f64).sqrt())
}

/// Orthonormal frame: the centre-of-mass direction, then the saddle
/// modes in the order of [`eigenvalue`]. The reflections after the first
/// fix its first column and mix the rest.
fn frame() -> Array2<f64> {
    let mut to_com = Array1::zeros(D);
    to_com[0] = 1.0;
    to_com -= &centre_of_mass();
    let mut q = reflection(&to_com);
    let mut rng = Rng(7);
    for _ in 0..3 {
        let mut w = rng.vector(D);
        w[0] = 0.0;
        q = q.dot(&reflection(&w));
    }
    q
}

fn mode(j: usize) -> Array1<f64> {
    frame().column(j + 1).to_owned()
}

fn saddle() -> Array1<f64> {
    -0.8 * &mode(0) + 0.5 * &mode(1) + 0.3 * &mode(2) + 0.4 * &centre_of_mass()
}

/// The free energy at the optimum, \(\tfrac{kT}{2}\ln|\lambda_1|\).
fn optimum_value() -> f64 {
    0.5 * KT * UNSTABLE.abs().ln()
}

struct Terms {
    /// Saddle relative to the pivot.
    q0: Array1<f64>,
    /// \(H^+ n\).
    m: Array1<f64>,
    s: f64,
    r: f64,
}

/// `f = -A` on packed `(n, d)`, with the pivot behind a lock so that a
/// gauge can move it while a driver holds the objective.
struct PlaneEnergy {
    pseudo_inverse: Array2<f64>,
    saddle: Array1<f64>,
    pivot: Mutex<Array1<f64>>,
    bounds: Bounds<f64>,
}

impl PlaneEnergy {
    fn new() -> Self {
        let q = frame();
        let mut inverse = Array1::zeros(D);
        for j in 0..D - 1 {
            inverse[j + 1] = 1.0 / eigenvalue(j);
        }
        Self {
            pseudo_inverse: q.dot(&Array2::from_diag(&inverse)).dot(&q.t()),
            saddle: saddle(),
            pivot: Mutex::new(Array1::zeros(D)),
            bounds: Bounds::new(
                Array1::from_elem(D + 1, -1e6),
                Array1::from_elem(D + 1, 1e6),
                0.0,
            ),
        }
    }

    fn pivot(&self) -> Array1<f64> {
        self.pivot.lock().unwrap().clone()
    }

    fn terms(&self, x: &Array1<f64>) -> Terms {
        let n = x.slice(s![..D]).to_owned();
        let q0 = &self.saddle - &self.pivot();
        let m = self.pseudo_inverse.dot(&n);
        Terms {
            s: n.dot(&m),
            r: x[D] - n.dot(&q0),
            q0,
            m,
        }
    }

    fn value(&self, x: &Array1<f64>) -> f64 {
        let t = self.terms(x);
        if t.s >= 0.0 {
            return f64::INFINITY;
        }
        -(0.5 * t.r * t.r / t.s + 0.5 * KT * (-t.s).ln())
    }

    fn egrad(&self, x: &Array1<f64>) -> Array1<f64> {
        let t = self.terms(x);
        let a = t.r / t.s;
        let grad_n = -a * &t.q0 + (KT / t.s - a * a) * &t.m;
        let mut g = Array1::zeros(D + 1);
        g.slice_mut(s![..D]).assign(&(-grad_n));
        g[D] = -a;
        g
    }

    fn ehess(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let t = self.terms(x);
        let u = v.slice(s![..D]).to_owned();
        let a = t.r / t.s;
        let ds = 2.0 * t.m.dot(&u);
        let da = (v[D] - t.q0.dot(&u) - a * ds) / t.s;
        let hess_n = -da * &t.q0
            + (-KT * ds / (t.s * t.s) - 2.0 * a * da) * &t.m
            + (KT / t.s - a * a) * &self.pseudo_inverse.dot(&u);
        let mut h = Array1::zeros(D + 1);
        h.slice_mut(s![..D]).assign(&(-hess_n));
        h[D] = -da;
        h
    }

    /// Moves the pivot to the in-plane minimum of the harmonic energy,
    /// \(q^\ddagger + (r/s) H^+ n\), and zeroes the offset. The plane and
    /// `f` stay.
    fn recentre(&self, x: &mut Array1<f64>) {
        let t = self.terms(x);
        *self.pivot.lock().unwrap() = &self.saddle + &(t.r / t.s * &t.m);
        x[D] = 0.0;
    }

    /// Offset of the plane from the origin, which no gauge changes.
    fn offset_from_origin(&self, x: &Array1<f64>) -> f64 {
        x[D] + x.slice(s![..D]).dot(&self.pivot())
    }
}

impl Objective<f64> for PlaneEnergy {
    fn dim(&self) -> usize {
        D + 1
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value(&x.to_owned())
    }
}

impl Gradient<f64> for PlaneEnergy {
    fn dim(&self) -> usize {
        D + 1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.egrad(&x.to_owned())
    }
}

impl DifferentiableObjective<f64> for PlaneEnergy {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let x = x.to_owned();
        (self.value(&x), self.egrad(&x))
    }
}

/// The negative harmonic Hessian of the rotation block at the saddle
/// plane, \(kT(I + |\lambda_1| H_\parallel^+)\), compressed to the tangent
/// space of `n`. It is fixed in the ambient space, so its compression
/// stays positive definite at every normal.
struct HarmonicRotation {
    frame: Array2<f64>,
    stiffness: Array1<f64>,
}

impl HarmonicRotation {
    fn new() -> Self {
        let mut stiffness = Array1::from_elem(D, KT);
        for j in 1..D - 1 {
            stiffness[j + 1] = KT * (1.0 + UNSTABLE.abs() / eigenvalue(j));
        }
        Self {
            frame: frame(),
            stiffness,
        }
    }

    fn ambient(&self, v: &Array1<f64>, power: f64) -> Array1<f64> {
        let scaled = self.frame.t().dot(v) * self.stiffness.mapv(|k| k.powf(power));
        self.frame.dot(&scaled)
    }

    fn tangent(n: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let com = centre_of_mass();
        v - &(n.dot(v) * n) - &(com.dot(v) * &com)
    }
}

impl TangentMetric for HarmonicRotation {
    fn apply(&self, n: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Self::tangent(n, &self.ambient(&Self::tangent(n, v), 1.0))
    }

    /// The tangent `y` with `P R y = r`: `y = R^-1 (r - C mu)` for the
    /// normal basis `C = [n, e]`, with `mu` from the 2x2 Schur complement
    /// so that `C^T y = 0`.
    fn solve(&self, n: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
        let normals = [n.clone(), centre_of_mass()];
        let inverse = normals.clone().map(|c| self.ambient(&c, -1.0));
        let target = self.ambient(r, -1.0);
        let schur = |i: usize, j: usize| normals[i].dot(&inverse[j]);
        let (b0, b1) = (normals[0].dot(&target), normals[1].dot(&target));
        let det = schur(0, 0) * schur(1, 1) - schur(0, 1) * schur(1, 0);
        let mu0 = (b0 * schur(1, 1) - schur(0, 1) * b1) / det;
        let mu1 = (schur(0, 0) * b1 - schur(1, 0) * b0) / det;
        target - &(mu0 * &inverse[0]) - &(mu1 * &inverse[1])
    }
}

fn normal_factor() -> ProductFactor {
    let excluded = centre_of_mass().insert_axis(ndarray::Axis(0));
    ProductFactor::new(ManifoldKind::sphere_excluding(excluded).unwrap(), D)
}

/// Rotation weight \(\ell^2\) and offset weight one.
fn weighted_factors(rotation: f64) -> [ProductFactor; 2] {
    [
        normal_factor().with_weight(rotation),
        ProductFactor::new(ManifoldKind::Euclidean, 1),
    ]
}

/// Section 4.2's block weight, \(\ell^2 = kT(1 + |\lambda_1|/\bar\lambda)
/// / |\lambda_1|\) with the typical in-plane eigenvalue one.
fn block_weight() -> f64 {
    KT * (1.0 + UNSTABLE.abs()) / UNSTABLE.abs()
}

fn harmonic_factors() -> [ProductFactor; 2] {
    [
        normal_factor().with_metric(Arc::new(HarmonicRotation::new())),
        ProductFactor::new(ManifoldKind::Euclidean, 1).with_weight(UNSTABLE.abs()),
    ]
}

/// A normal a few degrees from the unstable mode, inside the domain
/// \(s < 0\), whose width along the softest mode is ten degrees, and a
/// positive offset.
fn start() -> Array1<f64> {
    let mut n = &mode(0) + &(0.06 * &mode(1));
    for j in 10..16 {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        n += &(0.1 * sign * &mode(j));
    }
    let n = &n / n.dot(&n).sqrt();
    let mut x = Array1::zeros(D + 1);
    x.slice_mut(s![..D]).assign(&n);
    x[D] = 0.3;
    x
}

fn assert_optimal_plane(energy: &PlaneEnergy, x: &Array1<f64>, value: f64, what: &str) {
    let n = x.slice(s![..D]).to_owned();
    assert!((n.dot(&n).sqrt() - 1.0).abs() < 1e-13, "{what}: |n| != 1");
    assert!(
        n.dot(&centre_of_mass()).abs() < 1e-13,
        "{what}: n moves the centre of mass"
    );
    let overlap = n.dot(&mode(0));
    assert!(overlap > 1.0 - 1e-10, "{what}: n . e_1 = {overlap}");
    let offset = energy.offset_from_origin(x);
    let expected = mode(0).dot(&saddle());
    assert!(
        (offset - expected).abs() < 1e-6,
        "{what}: offset {offset}, expected {expected}"
    );
    assert!(
        (value - optimum_value()).abs() < 1e-10,
        "{what}: f = {value}"
    );
}

fn rtr(factors: [ProductFactor; 2], energy: &PlaneEnergy) -> RtrMin {
    let product = ProductManifold::new(factors).unwrap();
    minimize_rtr(
        &product,
        start(),
        |x| energy.value(x),
        |x| energy.egrad(x),
        |x, v| energy.ehess(x, v),
        RtrOptions::default().with_grad_tol(1e-10),
    )
}

#[test]
fn the_optimal_plane_has_a_negative_offset_from_a_fixed_pivot() {
    let energy = PlaneEnergy::new();
    let out = rtr(weighted_factors(block_weight()), &energy);
    assert!(out.grad_norm <= 1e-10, "|grad| = {}", out.grad_norm);
    assert_optimal_plane(&energy, &out.x, out.value, "Euclidean offset");
    assert!(out.x[D] < -0.5, "d = {}", out.x[D]);

    // On SPD(1), the positive half-line, the offset cannot change sign:
    // the retraction d + v + v^2 / (2d) never goes below d / 2, so the run
    // creeps toward zero and ends well short of the optimum.
    let product = ProductManifold::new([
        normal_factor().with_weight(block_weight()),
        ProductFactor::new(ManifoldKind::Spd, 1),
    ])
    .unwrap();
    let out = minimize_rtr(
        &product,
        start(),
        |x| energy.value(x),
        |x| energy.egrad(x),
        |x, v| energy.ehess(x, v),
        RtrOptions::default().with_grad_tol(1e-10),
    );
    assert!(out.x[D] > 0.0 && out.x[D] < 1e-3, "d = {}", out.x[D]);
    assert!(out.value > optimum_value() + 0.4, "f = {}", out.value);
}

#[test]
fn the_gradient_is_the_riesz_vector_of_the_harmonic_metric() {
    let energy = PlaneEnergy::new();
    let product = ProductManifold::new(harmonic_factors()).unwrap();
    let x = start();
    let egrad = energy.egrad(&x);
    let rgrad = product.egrad2rgrad(&x, &egrad);
    assert!(max_abs(&(&product.project(&x, &rgrad) - &rgrad)) < 1e-14);
    let mut rng = Rng(3);
    for _ in 0..3 {
        let w = product.project(&x, &rng.vector(D + 1));
        let df = egrad.dot(&w);
        assert!((product.inner(&x, &rgrad, &w) - df).abs() < 1e-12 * (1.0 + df.abs()));
    }
}

#[test]
fn a_harmonic_tangent_metric_preconditions_the_trust_region() {
    let energy = PlaneEnergy::new();
    let unweighted = rtr(weighted_factors(1.0), &energy);
    let weighted = rtr(weighted_factors(block_weight()), &energy);
    let harmonic = rtr(harmonic_factors(), &energy);
    for (out, what) in [
        (&unweighted, "unweighted"),
        (&weighted, "block weights"),
        (&harmonic, "harmonic metric"),
    ] {
        assert!(out.grad_norm <= 1e-10, "{what}: |grad| = {}", out.grad_norm);
        assert_optimal_plane(&energy, &out.x, out.value, what);
    }
    // Exact second derivatives keep the outer counts close. The metric
    // shows in the inner solves: at the optimum the harmonic metric makes
    // the Hessian the identity plus a rank-two coupling of rotation and
    // offset.
    assert!(
        2 * harmonic.inner_iterations < weighted.inner_iterations,
        "{} inner iterations against {} with block weights",
        harmonic.inner_iterations,
        weighted.inner_iterations
    );
    assert!(weighted.inner_iterations <= unweighted.inner_iterations);
    assert!(harmonic.iterations <= weighted.iterations);
}

#[test]
fn a_sampled_trust_region_takes_its_steps_from_the_tangent_subproblem() {
    // The loop of a sampled driver: the step from the model, the decrease
    // measured outside it (by BAR in the sampler, exactly here), and the
    // radius from their ratio.
    let energy = PlaneEnergy::new();
    let product = ProductManifold::new(harmonic_factors()).unwrap();
    let opts = RtrOptions::default();
    let mut radius = RtrRadius::new(product.typical_dist(D + 1));
    let mut x = start();
    let mut value = energy.value(&x);
    for _ in 0..100 {
        let egrad = energy.egrad(&x);
        if product.norm(&x, &product.egrad2rgrad(&x, &egrad)) < 1e-10 {
            break;
        }
        let step = truncated_cg_riemannian(
            &product,
            &x,
            &egrad,
            |v| energy.ehess(&x, v),
            radius.radius,
            &opts,
        );
        let trial = product.retract(&x, &step.eta);
        let trial_value = energy.value(&trial);
        let eta_norm = product.norm(&x, &step.eta);
        if radius.update(value - trial_value, step.model_decrease, eta_norm) {
            x = trial;
            value = trial_value;
        }
    }
    assert_optimal_plane(&energy, &x, value, "sampled loop");
}

#[test]
fn recentring_the_pivot_moves_the_representation_not_the_plane() {
    let fixed = PlaneEnergy::new();
    let reference = rtr(harmonic_factors(), &fixed);
    assert_optimal_plane(&fixed, &reference.x, reference.value, "fixed pivot");

    let energy = PlaneEnergy::new();
    let product = ProductManifold::new(harmonic_factors()).unwrap();
    let mut moves = 0;
    let out = minimize_rtr_gauged(
        &product,
        start(),
        |x| energy.value(x),
        |x| energy.egrad(x),
        |x, v| energy.ehess(x, v),
        |x| {
            moves += 1;
            energy.recentre(x);
        },
        RtrOptions::default().with_grad_tol(1e-10),
    );
    assert!(out.grad_norm <= 1e-10, "|grad| = {}", out.grad_norm);
    assert_eq!(moves, out.accepted);
    assert_eq!(out.x[D], 0.0);
    assert_optimal_plane(&energy, &out.x, out.value, "re-centred pivot");
    // The in-plane centroid of the optimal plane is the saddle.
    assert!(max_abs(&(&energy.pivot() - &saddle())) < 1e-7);
    // About the centroid the offset decouples from the rotation, the
    // Hessian is the harmonic metric itself, and the inner solves shorten.
    assert!(
        out.inner_iterations < reference.inner_iterations,
        "{} inner iterations against {} about a fixed pivot",
        out.inner_iterations,
        reference.inner_iterations
    );
}

#[test]
fn a_session_moves_to_the_gauge_between_steps() {
    let energy = PlaneEnergy::new();
    let control = Control::default().with_maxiter(500).with_gtol(1e-7);
    let mut solver = Solver::new(Method::lbfgs(), control, D + 1);
    solver
        .set_product(weighted_factors(block_weight()))
        .unwrap();
    // The free energy is finite only where the plane cuts the saddle, so
    // the line search backtracks from trial planes past that edge.
    solver.set_accept(Accept::Energy);
    let mut x = start();
    let (mut calls, mut moves) = (0, 0);
    let report = solver
        .run_gauged(&energy, &mut x, |x| {
            calls += 1;
            let offset = x[D];
            energy.recentre(x);
            if x[D] != offset {
                moves += 1;
            }
        })
        .unwrap();
    assert!(report.grad_norm < 1e-7, "|grad| = {}", report.grad_norm);
    assert_eq!(calls + 1, solver.steps());
    assert!(moves > 0);
    assert_eq!(report.coords, x);
    assert_optimal_plane(&energy, &x, report.value, "L-BFGS, re-centred pivot");
}

#[test]
fn sessions_refuse_a_product_with_a_tangent_metric() {
    let energy = PlaneEnergy::new();
    for method in [Method::lbfgs(), Method::Steepest, Method::NewtonCg] {
        let mut solver = Solver::new(method, Control::default(), D + 1);
        solver.set_product(harmonic_factors()).unwrap();
        let mut x = start();
        let err = solver.step(&energy, &mut x).unwrap_err();
        assert!(
            matches!(err, Error::ProductTangentMetric { index: 0 }),
            "{err}"
        );
        assert_eq!(x, start());
        let g = energy.egrad(&x);
        assert!(!solver.push_pair(g.view(), g.view()));
        assert!(matches!(
            solver.search_direction(g.view()),
            Err(Error::ProductTangentMetric { index: 0 })
        ));
    }
}
