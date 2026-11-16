//! The unit sphere of a linear subspace. manopt `checkgradient` and
//! `checkhessian` slopes, the geometry under both constructions, the
//! Hessian across extensions of the cost, and the drivers on a Rayleigh
//! quotient whose lowest mode overall lies outside the subspace.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array, s};
use rgmin::manifold::{Manifold, Sphere, SubspaceSphere};
use rgmin::rtr::{RtrOptions, minimize_rtr};
use rgmin::{Control, ManifoldKind, Method, Report, Solver};

const D: usize = 6;

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

/// Orthogonal frame of \(\mathbb{R}^6\) with no coordinate axis among its
/// columns. The first two span the excluded plane.
fn frame() -> Array2<f64> {
    reflection(&array![1.0, -2.0, 0.5, 1.5, 0.3, -0.7])
        .dot(&reflection(&array![0.4, 0.1, -1.2, 0.8, 2.0, 0.6]))
}

fn excluded() -> Array2<f64> {
    frame().slice(s![.., ..2]).t().to_owned()
}

fn kept() -> Array2<f64> {
    frame().slice(s![.., 2..]).t().to_owned()
}

fn subspace_sphere() -> SubspaceSphere {
    SubspaceSphere::excluding(excluded()).unwrap()
}

/// Largest component of `v` along the excluded directions.
fn off_subspace(v: &Array1<f64>) -> f64 {
    max_abs(&excluded().dot(v))
}

fn unit_tangent<M: Manifold>(m: &M, x: &Array1<f64>, rng: &mut Rng) -> Array1<f64> {
    let v = m.project(x, &rng.vector(x.len()));
    let norm = m.norm(x, &v);
    v / norm
}

/// `f(x) = x^T A x / 2 + b^T x + sum x_i^4 / 4`, with a dense `A`.
struct Quartic {
    a: Array2<f64>,
    b: Array1<f64>,
}

impl Quartic {
    fn new(rng: &mut Rng) -> Self {
        let m = Array2::from_shape_fn((D, D), |_| rng.next());
        Self {
            a: &m + &m.t(),
            b: rng.vector(D),
        }
    }

    fn value(&self, x: &Array1<f64>) -> f64 {
        0.5 * x.dot(&self.a.dot(x))
            + self.b.dot(x)
            + 0.25 * x.iter().map(|v| v.powi(4)).sum::<f64>()
    }

    fn egrad(&self, x: &Array1<f64>) -> Array1<f64> {
        self.a.dot(x) + &self.b + x.mapv(|v| v.powi(3))
    }

    fn ehess(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.a.dot(v) + 3.0 * &(x.mapv(|t| t * t) * v)
    }
}

/// Least-squares slope of `log10 |e(t)|` against `log10 t` for `t` from
/// `10^-2.5` down to `10^-4`. manopt reads a decade higher; there the
/// normal part of this cost's gradient, of order ten, still bends the
/// remainder with the cubic term.
fn slope(e: impl Fn(f64) -> f64) -> f64 {
    let points: Vec<(f64, f64)> = (0..7)
        .map(|k| {
            let lt = -2.5 - 0.25 * f64::from(k);
            (lt, e(10f64.powf(lt)).abs().log10())
        })
        .collect();
    let count = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / count;
    let my = points.iter().map(|p| p.1).sum::<f64>() / count;
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    sxy / sxx
}

#[test]
fn subspace_sphere_passes_checkgradient_and_checkhessian() {
    let m = subspace_sphere();
    let mut rng = Rng(11);
    let f = Quartic::new(&mut rng);
    for _ in 0..3 {
        let x = m.closest_point(&rng.vector(D)).unwrap();
        let v = unit_tangent(&m, &x, &mut rng);
        let f0 = f.value(&x);
        let egrad = f.egrad(&x);
        let rgrad = m.egrad2rgrad(&x, &egrad);
        assert!(max_abs(&(&m.project(&x, &rgrad) - &rgrad)) < 1e-14);
        let df = m.inner(&x, &rgrad, &v);
        assert!((df - egrad.dot(&v)).abs() < 1e-12);
        let h = m.ehess2rhess(&x, &egrad, &f.ehess(&x, &v), &v).unwrap();
        assert!(max_abs(&(&m.project(&x, &h) - &h)) < 1e-14);
        let d2f = m.inner(&x, &h, &v);
        // Both curves are second order, so the Hessian model leaves a
        // cubic remainder along either.
        for exponential in [false, true] {
            let along = |t: f64| {
                let tv = t * &v;
                if exponential {
                    m.exp(&x, &tv).unwrap()
                } else {
                    m.retract(&x, &tv)
                }
            };
            let first = slope(|t| f.value(&along(t)) - f0 - t * df);
            let second = slope(|t| f.value(&along(t)) - f0 - t * df - 0.5 * t * t * d2f);
            assert!((first - 2.0).abs() < 0.1, "gradient slope {first}");
            assert!((second - 3.0).abs() < 0.1, "Hessian slope {second}");
        }
        let u = unit_tangent(&m, &x, &mut rng);
        let hu = m.ehess2rhess(&x, &egrad, &f.ehess(&x, &u), &u).unwrap();
        assert!((m.inner(&x, &hu, &v) - m.inner(&x, &u, &h)).abs() < 1e-12);
    }
}

#[test]
fn a_kept_basis_and_its_excluded_complement_give_one_sphere() {
    let within = SubspaceSphere::within(kept()).unwrap();
    let excluding = subspace_sphere();
    assert_eq!(within.subspace_dim(), 4);
    assert_eq!(excluding.subspace_dim(), 4);
    let mut rng = Rng(5);
    for _ in 0..5 {
        let v = rng.vector(D);
        let pv = within.project_subspace(&v);
        assert!(max_abs(&(&pv - &excluding.project_subspace(&v))) < 1e-14);
        let x = within.closest_point(&v).unwrap();
        assert!(max_abs(&(&x - &excluding.closest_point(&v).unwrap())) < 1e-14);
        let w = rng.vector(D);
        let t = within.project(&x, &w);
        assert!(max_abs(&(&t - &excluding.project(&x, &w))) < 1e-14);
        let y = within.retract(&x, &t);
        assert!(max_abs(&(&y - &excluding.retract(&x, &t))) < 1e-14);
    }
}

#[test]
fn steps_and_transports_stay_on_the_subspace_sphere() {
    let m = subspace_sphere();
    let mut rng = Rng(23);
    for _ in 0..5 {
        let x = m.closest_point(&rng.vector(D)).unwrap();
        let v = 0.8 * &unit_tangent(&m, &x, &mut rng);
        for y in [m.retract(&x, &v), m.exp(&x, &v).unwrap()] {
            assert!((y.dot(&y).sqrt() - 1.0).abs() < 1e-14);
            assert!(off_subspace(&y) < 1e-14);
        }
        let y = m.exp(&x, &v).unwrap();
        assert!(max_abs(&(&m.log(&x, &y).unwrap() - &v)) < 1e-12);
        assert!((m.dist(&x, &y).unwrap() - 0.8).abs() < 1e-12);

        let w = unit_tangent(&m, &x, &mut rng);
        let moved = m.transport(&x, &y, &w);
        assert!(moved.dot(&y).abs() < 1e-14 && off_subspace(&moved) < 1e-14);
        let carried = m.parallel_transport_along(&x, &v, &w).unwrap();
        assert!(carried.dot(&y).abs() < 1e-14 && off_subspace(&carried) < 1e-14);
        assert!((m.norm(&y, &carried) - 1.0).abs() < 1e-14);
        // The velocity of the geodesic is parallel along it.
        let velocity = m.parallel_transport_along(&x, &v, &v).unwrap();
        let back = m.log(&y, &x).unwrap();
        assert!(max_abs(&(&velocity + &back)) < 1e-12);
    }
    // A start off the sphere or out of the subspace lands on the closest
    // point of the sphere.
    let x = array![1.0, 2.0, -1.0, 0.5, 0.0, 3.0];
    let y = m.retract(&x, &Array1::zeros(D));
    assert!((y.dot(&y).sqrt() - 1.0).abs() < 1e-14 && off_subspace(&y) < 1e-14);
}

/// The Riemannian Hessian of `f` at `x` from the extension `f + h`,
/// with `h(x) = (|x|^2 - 1)(c^T x + 2)` vanishing on the sphere.
fn shifted_hessian<M: Manifold>(
    m: &M,
    f: &Quartic,
    c: &Array1<f64>,
    x: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    let lift = c.dot(x) + 2.0;
    let egrad = f.egrad(x) + 2.0 * lift * x + (x.dot(x) - 1.0) * c;
    let ehess = f.ehess(x, v) + 2.0 * lift * v + 2.0 * c.dot(v) * x + 2.0 * x.dot(v) * c;
    m.ehess2rhess(x, &egrad, &ehess, v).unwrap()
}

/// Gradient of the degree-zero extension `f(x / |x|)`.
fn homogeneous_gradient(f: &Quartic, x: &Array1<f64>) -> Array1<f64> {
    let r = x.dot(x).sqrt();
    let u = x / r;
    let g = f.egrad(&u);
    (&g - g.dot(&u) * &u) / r
}

fn check_extensions<M: Manifold>(m: &M, x: &Array1<f64>, v: &Array1<f64>, f: &Quartic) {
    let c = array![0.3, -1.1, 0.4, 0.9, -0.2, 0.6];
    let hess = m.ehess2rhess(x, &f.egrad(x), &f.ehess(x, v), v).unwrap();
    let shifted = shifted_hessian(m, f, &c, x, v);
    assert!(max_abs(&(&shifted - &hess)) < 1e-12);
    // The projected Hessians of the two extensions differ by
    // 2 (c^T x + 2) v, which the Weingarten term removes.
    let projected = m.project(x, &f.ehess(x, v));
    assert!(max_abs(&(&projected - &hess)) > 0.1);

    let g0 = homogeneous_gradient(f, x);
    assert!(g0.dot(x).abs() < 1e-14);
    let h = 1e-5;
    let ehess0 = (homogeneous_gradient(f, &(x + &(h * v)))
        - homogeneous_gradient(f, &(x - &(h * v))))
        / (2.0 * h);
    assert!(max_abs(&(&m.project(x, &ehess0) - &hess)) < 1e-8);
    let weingarten = m.ehess2rhess(x, &g0, &ehess0, v).unwrap();
    assert!(max_abs(&(&weingarten - &hess)) < 1e-8);
}

#[test]
fn the_hessian_does_not_depend_on_the_extension() {
    let mut rng = Rng(31);
    let f = Quartic::new(&mut rng);
    let m = subspace_sphere();
    for _ in 0..3 {
        let x = m.closest_point(&rng.vector(D)).unwrap();
        let v = unit_tangent(&m, &x, &mut rng);
        check_extensions(&m, &x, &v, &f);
        let x = &x / x.dot(&x).sqrt();
        let v = unit_tangent(&Sphere, &x, &mut rng);
        check_extensions(&Sphere, &x, &v, &f);
    }
}

/// `x^T A x` on the sphere, or its degree-zero extension
/// `x^T A x / x^T x`. `A` couples the subspace to the excluded plane,
/// which holds the two lowest modes, but its block on the subspace is
/// diagonal in the frame with lowest eigenvalue one.
struct Rayleigh {
    a: Array2<f64>,
    homogeneous: bool,
    bounds: Bounds<f64>,
}

const SUBSPACE_LOWEST: f64 = 1.0;

impl Rayleigh {
    fn new(homogeneous: bool) -> Self {
        let q = frame();
        let mut core = Array2::from_diag(&array![-3.0, -2.0, SUBSPACE_LOWEST, 2.0, 3.5, 5.0]);
        for (e, k) in [(0, 2), (1, 3), (0, 5)] {
            core[(e, k)] = 0.7;
            core[(k, e)] = 0.7;
        }
        Self {
            a: q.dot(&core).dot(&q.t()),
            homogeneous,
            bounds: Bounds::new(Array1::from_elem(D, -1e6), Array1::from_elem(D, 1e6), 0.0),
        }
    }

    fn lowest_mode() -> Array1<f64> {
        frame().column(2).to_owned()
    }

    fn value_of(&self, x: ArrayView1<f64>) -> f64 {
        let q = x.dot(&self.a.dot(&x));
        if self.homogeneous { q / x.dot(&x) } else { q }
    }

    fn gradient_of(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let ax = self.a.dot(&x);
        if self.homogeneous {
            let xx = x.dot(&x);
            2.0 * (&ax - &(self.value_of(x) * &x)) / xx
        } else {
            2.0 * ax
        }
    }
}

impl Objective<f64> for Rayleigh {
    fn dim(&self) -> usize {
        D
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_of(x)
    }
}

impl Gradient<f64> for Rayleigh {
    fn dim(&self) -> usize {
        D
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradient_of(x)
    }
}

impl DifferentiableObjective<f64> for Rayleigh {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.value_of(x), self.gradient_of(x))
    }
}

fn rayleigh_start() -> Array1<f64> {
    subspace_sphere()
        .closest_point(&array![0.4, 1.0, -0.3, 0.8, 0.2, -0.5])
        .unwrap()
}

fn solve(method: Method, kind: ManifoldKind, problem: &Rayleigh) -> (Array1<f64>, Report) {
    let control = Control::default().with_maxiter(500).with_gtol(1e-10);
    let mut solver = Solver::new(method, control, D);
    solver.set_manifold(kind);
    let mut x = rayleigh_start();
    let report = solver.run(problem, &mut x).unwrap();
    (x, report)
}

fn assert_lowest_in_subspace(x: &Array1<f64>, value: f64, what: &str) {
    assert!(
        (value - SUBSPACE_LOWEST).abs() < 1e-9,
        "{what}: f = {value}"
    );
    assert!(off_subspace(x) < 1e-13, "{what}: left the subspace");
    assert!(
        (x.dot(x).sqrt() - 1.0).abs() < 1e-13,
        "{what}: left the sphere"
    );
    let overlap = x.dot(&Rayleigh::lowest_mode()).abs();
    assert!(overlap > 1.0 - 1e-9, "{what}: |x . e| = {overlap}");
}

#[test]
fn the_drivers_find_the_lowest_mode_inside_the_subspace() {
    let kind = ManifoldKind::sphere_excluding(excluded()).unwrap();
    let plain = Rayleigh::new(false);
    let (x, report) = solve(Method::lbfgs(), kind.clone(), &plain);
    assert!(report.grad_norm < 1e-10);
    assert_lowest_in_subspace(&x, report.value, "L-BFGS");
    // The whole sphere reaches the excluded modes below.
    let (_, whole) = solve(Method::lbfgs(), ManifoldKind::Sphere, &plain);
    assert!(whole.value < -2.0, "whole sphere: f = {}", whole.value);

    let out = minimize_rtr(
        &subspace_sphere(),
        rayleigh_start(),
        |x| plain.value_of(x.view()),
        |x| plain.gradient_of(x.view()),
        |_, v| 2.0 * plain.a.dot(v),
        RtrOptions::default().with_grad_tol(1e-12),
    );
    assert!(out.grad_norm <= 1e-12);
    assert_lowest_in_subspace(&out.x, out.value, "RTR");
    assert!(out.iterations < 15, "{} RTR iterations", out.iterations);
}

/// Gradient norms of session Newton-CG steps down to `1e-7`, above the
/// level where the decrease of `f` is lost to rounding.
fn newton_cg_gradients(problem: &Rayleigh, what: &str) -> Vec<f64> {
    let control = Control::default().with_maxiter(100).with_gtol(1e-7);
    let mut solver = Solver::new(Method::NewtonCg, control, D);
    solver.set_manifold(ManifoldKind::sphere_excluding(excluded()).unwrap());
    let mut x = rayleigh_start();
    let mut norms = Vec::new();
    loop {
        let report = solver.step(problem, &mut x).unwrap();
        norms.push(report.grad_norm);
        if report.grad_norm < 1e-7 || norms.len() == 100 {
            assert_lowest_in_subspace(&x, report.value, what);
            return norms;
        }
    }
}

#[test]
fn session_newton_cg_is_exact_for_the_degree_zero_extension() {
    let exact = newton_cg_gradients(&Rayleigh::new(true), "degree-zero extension");
    assert!(exact.len() <= 10, "{} steps", exact.len());
    // The session model omits the Weingarten term, so for x^T A x it
    // overstates the tangent Hessian 2 (A - f) by 2 f = 2. Near the
    // lowest mode each step then keeps lowest / next = 1 / 2 of the
    // gradient along the next mode.
    let plain = newton_cg_gradients(&Rayleigh::new(false), "plain extension");
    let rates: Vec<f64> = plain
        .windows(2)
        .filter(|w| w[0] < 1e-3)
        .map(|w| w[1] / w[0])
        .collect();
    assert!(rates.len() > 5, "{plain:?}");
    assert!(
        rates.iter().all(|r| (r - 0.5).abs() < 0.02),
        "rates {rates:?}"
    );
}
