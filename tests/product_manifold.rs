//! The weighted product manifold. manopt `checkgradient` and
//! `checkhessian` slopes along retraction and exponential curves, the
//! geometry of the blocks, and the session and trust-region drivers on a
//! coupled Rayleigh quotient and quadratic on \(\mathbb{R}^n \times
//! S^{n-1}\), the optimal-hyperplane geometry.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array, s};
use rgmin::manifold::{ProductFactor, ProductManifold};
use rgmin::rtr::{RtrMin, RtrOptions, minimize_rtr};
use rgmin::{
    Accept, Control, Error, HessianVector, Manifold, ManifoldKind, Method, Report, Solver,
};

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

fn join(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    a.iter().chain(b.iter()).copied().collect()
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0, |m, e| m.max(e.abs()))
}

/// `f(x) = x^T A x / 2 + b^T x + sum x_i^4 / 4`. The dense symmetric `A`
/// couples every block.
struct Quartic {
    a: Array2<f64>,
    b: Array1<f64>,
}

impl Quartic {
    fn new(n: usize, rng: &mut Rng) -> Self {
        let m = Array2::from_shape_fn((n, n), |_| rng.next());
        Self {
            a: &m + &m.t(),
            b: rng.vector(n),
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

/// \(\mathbb{R}^4 \times S^3\) with a position weight in inverse squared
/// length and a normal weight of seven.
fn hyperplane_factors(position_weight: f64) -> [ProductFactor; 2] {
    [
        ProductFactor::new(ManifoldKind::Euclidean, N).with_weight(position_weight),
        ProductFactor::new(ManifoldKind::Sphere, N).with_weight(7.0),
    ]
}

fn hyperplane() -> ProductManifold {
    ProductManifold::new(hyperplane_factors(POSITION_WEIGHT)).unwrap()
}

fn sphere_by_rotation() -> ProductManifold {
    ProductManifold::new([
        ProductFactor::new(ManifoldKind::Sphere, 3).with_weight(2.5),
        ProductFactor::new(ManifoldKind::So3, 9).with_weight(0.4),
    ])
    .unwrap()
}

fn base_point(kind: &ManifoldKind, len: usize) -> Array1<f64> {
    let mut x = Array1::zeros(len);
    match kind {
        ManifoldKind::Sphere | ManifoldKind::Stiefel | ManifoldKind::SphereComplex { .. } => {
            x[0] = 1.0;
        }
        ManifoldKind::So3 | ManifoldKind::Se3 => {
            x[0] = 1.0;
            x[4] = 1.0;
            x[8] = 1.0;
        }
        ManifoldKind::Oblique { n, m } => {
            for j in 0..*m {
                x[j * n] = 1.0;
            }
        }
        ManifoldKind::ComplexCircle { .. } => {
            for j in (0..len).step_by(2) {
                x[j] = 1.0;
            }
        }
        ManifoldKind::Positive { .. } => x.fill(1.0),
        _ => {}
    }
    x
}

fn random_point(product: &ProductManifold, rng: &mut Rng) -> Array1<f64> {
    let mut base = Array1::zeros(product.packed_len());
    for (i, factor) in product.factors().iter().enumerate() {
        let range = product.range(i).unwrap();
        base.slice_mut(s![range])
            .assign(&base_point(&factor.kind, factor.len));
    }
    let step = product.project(&base, &(0.5 * &rng.vector(base.len())));
    product.retract(&base, &step)
}

/// A tangent vector at `x` of unit length in the product metric.
fn unit_tangent(product: &ProductManifold, x: &Array1<f64>, rng: &mut Rng) -> Array1<f64> {
    let v = product.project(x, &rng.vector(x.len()));
    let norm = product.norm(x, &v);
    v / norm
}

/// Largest normal component of `v` at `x`, relative to `v`.
fn normal_part(product: &ProductManifold, x: &Array1<f64>, v: &Array1<f64>) -> f64 {
    max_abs(&(&product.project(x, v) - v)) / (1.0 + max_abs(v))
}

/// Largest constraint violation of the sphere and rotation blocks.
fn off_manifold(product: &ProductManifold, x: &Array1<f64>) -> f64 {
    let mut worst: f64 = 0.0;
    for (i, factor) in product.factors().iter().enumerate() {
        let b = x.slice(s![product.range(i).unwrap()]);
        let violation = match factor.kind {
            ManifoldKind::Sphere => (b.dot(&b).sqrt() - 1.0).abs(),
            ManifoldKind::So3 => {
                let r = Array2::from_shape_vec((3, 3), b.to_vec()).unwrap();
                let gram = r.t().dot(&r) - Array2::<f64>::eye(3);
                let det = r[(0, 0)] * (r[(1, 1)] * r[(2, 2)] - r[(1, 2)] * r[(2, 1)])
                    - r[(0, 1)] * (r[(1, 0)] * r[(2, 2)] - r[(1, 2)] * r[(2, 0)])
                    + r[(0, 2)] * (r[(1, 0)] * r[(2, 1)] - r[(1, 1)] * r[(2, 0)]);
                gram.iter().fold((det - 1.0).abs(), |m, v| m.max(v.abs()))
            }
            _ => 0.0,
        };
        worst = worst.max(violation);
    }
    worst
}

#[derive(Clone, Copy, Debug)]
enum Curve {
    Retraction,
    Exponential,
}

fn along(
    product: &ProductManifold,
    x: &Array1<f64>,
    v: &Array1<f64>,
    how: Curve,
    t: f64,
) -> Array1<f64> {
    let tv = t * v;
    match how {
        Curve::Retraction => product.retract(x, &tv),
        Curve::Exponential => product.exp(x, &tv).expect("closed-form exponential"),
    }
}

/// Least-squares slope of `log10 |e(t)|` against `log10 t` for `t` from
/// `10^-1.5` down to `10^-3`, the window manopt reads from its plot.
fn slope(e: impl Fn(f64) -> f64) -> f64 {
    let points: Vec<(f64, f64)> = (0..7)
        .map(|k| {
            let lt = -1.5 - 0.25 * f64::from(k);
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

/// manopt `checkgradient`: the remainder of the first-order model along
/// the curve. Also checks that the Riemannian gradient is tangent and
/// represents the differential in the product metric.
fn gradient_slope(
    product: &ProductManifold,
    f: &Quartic,
    x: &Array1<f64>,
    v: &Array1<f64>,
    how: Curve,
) -> f64 {
    let f0 = f.value(x);
    let egrad = f.egrad(x);
    let rgrad = product.egrad2rgrad(x, &egrad);
    let df = product.inner(x, &rgrad, v);
    assert!(
        (df - egrad.dot(v)).abs() <= 1e-12 * (1.0 + df.abs()),
        "<grad f, v> = {df} but Df[v] = {}",
        egrad.dot(v)
    );
    assert!(normal_part(product, x, &rgrad) < 1e-12);
    slope(|t| f.value(&along(product, x, v, how, t)) - f0 - t * df)
}

/// manopt `checkhessian`: the remainder of the second-order model along
/// the curve. Also checks that the Hessian action is tangent.
fn hessian_slope(
    product: &ProductManifold,
    f: &Quartic,
    x: &Array1<f64>,
    v: &Array1<f64>,
    how: Curve,
) -> f64 {
    let f0 = f.value(x);
    let egrad = f.egrad(x);
    let df = egrad.dot(v);
    let h = product
        .ehess2rhess(x, &egrad, &f.ehess(x, v), v)
        .expect("every factor supplies its Weingarten term");
    assert!(normal_part(product, x, &h) < 1e-12);
    let d2f = product.inner(x, &h, v);
    slope(|t| f.value(&along(product, x, v, how, t)) - f0 - t * df - 0.5 * t * t * d2f)
}

/// `<Hess f[u], v> - <u, Hess f[v]>` in the product metric.
fn hessian_asymmetry(
    product: &ProductManifold,
    f: &Quartic,
    x: &Array1<f64>,
    u: &Array1<f64>,
    v: &Array1<f64>,
) -> f64 {
    let egrad = f.egrad(x);
    let hu = product.ehess2rhess(x, &egrad, &f.ehess(x, u), u).unwrap();
    let hv = product.ehess2rhess(x, &egrad, &f.ehess(x, v), v).unwrap();
    (product.inner(x, &hu, v) - product.inner(x, u, &hv)).abs()
}

fn check_derivatives(product: &ProductManifold, hessian_curves: &[Curve], seed: u64, what: &str) {
    let mut rng = Rng(seed);
    let f = Quartic::new(product.packed_len(), &mut rng);
    for _ in 0..3 {
        let x = random_point(product, &mut rng);
        let v = unit_tangent(product, &x, &mut rng);
        for how in [Curve::Retraction, Curve::Exponential] {
            let g = gradient_slope(product, &f, &x, &v, how);
            assert!(
                (g - 2.0).abs() < 0.1,
                "{what}: gradient slope {g} along the {how:?}"
            );
        }
        for &how in hessian_curves {
            let h = hessian_slope(product, &f, &x, &v, how);
            assert!(
                (h - 3.0).abs() < 0.1,
                "{what}: Hessian slope {h} along the {how:?}"
            );
        }
        let u = unit_tangent(product, &x, &mut rng);
        let asymmetry = hessian_asymmetry(product, &f, &x, &u, &v);
        assert!(asymmetry < 1e-10, "{what}: Hessian asymmetry {asymmetry}");
    }
}

#[test]
fn weighted_hyperplane_passes_checkgradient_and_checkhessian() {
    // The straight line and the normalized sphere step are second-order
    // retractions, so the Hessian slope holds along them as well.
    check_derivatives(
        &hyperplane(),
        &[Curve::Retraction, Curve::Exponential],
        7,
        "R^4 x S^3",
    );
}

#[test]
fn sphere_by_rotation_passes_checkgradient_and_checkhessian() {
    // The QR retraction of SO(3) is first order. As in manopt, the Hessian
    // is checked along the exponential map, a second-order retraction.
    check_derivatives(
        &sphere_by_rotation(),
        &[Curve::Exponential],
        11,
        "S^2 x SO(3)",
    );
}

#[test]
fn every_factor_with_a_closed_form_geodesic_passes_checkhessian() {
    let factors = [
        (ManifoldKind::Euclidean, 3),
        (ManifoldKind::Sphere, 4),
        (ManifoldKind::Stiefel, 3),
        (ManifoldKind::So3, 9),
        (ManifoldKind::Se3, 12),
        (ManifoldKind::Oblique { n: 3, m: 2 }, 6),
        (ManifoldKind::Symmetric, 9),
        (ManifoldKind::SkewSymmetric, 9),
        (ManifoldKind::ComplexCircle { n: 2 }, 4),
        (ManifoldKind::EuclideanComplex { n: 2 }, 4),
        (ManifoldKind::SphereComplex { n: 2 }, 4),
        (ManifoldKind::Positive { n: 3 }, 3),
        (
            ManifoldKind::CenteredMatrix {
                m: 2,
                n: 3,
                rows: true,
            },
            6,
        ),
    ];
    for (seed, (kind, len)) in (100..).zip(factors) {
        let what = kind.as_str();
        let product = ProductManifold::new([
            ProductFactor::new(kind.clone(), len).with_weight(3.0),
            ProductFactor::new(ManifoldKind::Euclidean, 2).with_weight(0.5),
        ])
        .unwrap();
        check_derivatives(&product, &[Curve::Exponential], seed, what);
    }
}

#[test]
fn retraction_and_exponential_stay_on_the_manifold() {
    let mut rng = Rng(3);
    for product in [hyperplane(), sphere_by_rotation()] {
        for _ in 0..4 {
            let x = random_point(&product, &mut rng);
            let v = unit_tangent(&product, &x, &mut rng);
            for length in [1e-3, 0.1, 1.0, 10.0] {
                let step = length * &v;
                let y = product.retract(&x, &step);
                assert!(
                    off_manifold(&product, &y) < 1e-12,
                    "retraction of length {length}"
                );
                let y = product.exp(&x, &step).unwrap();
                assert!(
                    off_manifold(&product, &y) < 1e-12,
                    "exponential of length {length}"
                );
            }
        }
    }
}

#[test]
fn transports_land_in_the_target_tangent_space() {
    let mut rng = Rng(5);
    for product in [hyperplane(), sphere_by_rotation()] {
        assert!(!product.transport_is_isometric());
        for _ in 0..4 {
            let x = random_point(&product, &mut rng);
            let d = 0.7 * &unit_tangent(&product, &x, &mut rng);
            let u = unit_tangent(&product, &x, &mut rng);
            let v = unit_tangent(&product, &x, &mut rng);
            for y in [product.retract(&x, &d), product.exp(&x, &d).unwrap()] {
                let w = product.transport(&x, &y, &v);
                assert!(normal_part(&product, &y, &w) < 1e-12);
            }
            let y = product.exp(&x, &d).unwrap();
            let pu = product.parallel_transport_along(&x, &d, &u).unwrap();
            let pv = product.parallel_transport_along(&x, &d, &v).unwrap();
            assert!(normal_part(&product, &y, &pu) < 1e-12);
            let before = product.inner(&x, &u, &v);
            let after = product.inner(&y, &pu, &pv);
            assert!((before - after).abs() < 1e-12, "{before} -> {after}");
        }
    }
}

#[test]
fn transport_of_flat_factors_is_an_isometry() {
    let product = ProductManifold::new([
        ProductFactor::new(ManifoldKind::Euclidean, 3).with_weight(2.0),
        ProductFactor::new(ManifoldKind::Symmetric, 9).with_weight(0.25),
        ProductFactor::new(ManifoldKind::SkewSymmetric, 9),
    ])
    .unwrap();
    assert!(product.transport_is_isometric());
    let mut rng = Rng(13);
    let x = random_point(&product, &mut rng);
    let y = product.retract(&x, &unit_tangent(&product, &x, &mut rng));
    let u = unit_tangent(&product, &x, &mut rng);
    let v = unit_tangent(&product, &x, &mut rng);
    let tu = product.transport(&x, &y, &u);
    let tv = product.transport(&x, &y, &v);
    assert!((product.inner(&x, &u, &v) - product.inner(&y, &tu, &tv)).abs() < 1e-12);
}

#[test]
fn parallel_transport_carries_the_geodesic_velocity() {
    // Huang, Gallivan and Absil's locking condition for the exponential
    // map and parallel transport, with beta = 1: the velocity of
    // t -> exp_x(t d) at t = 1 is the parallel transport of d.
    let mut rng = Rng(17);
    let h = 1e-5;
    for product in [hyperplane(), sphere_by_rotation()] {
        for _ in 0..3 {
            let x = random_point(&product, &mut rng);
            let d = 0.8 * &unit_tangent(&product, &x, &mut rng);
            let ahead = product.exp(&x, &((1.0 + h) * &d)).unwrap();
            let behind = product.exp(&x, &((1.0 - h) * &d)).unwrap();
            let velocity = (&ahead - &behind) / (2.0 * h);
            let carried = product.parallel_transport_along(&x, &d, &d).unwrap();
            assert!(max_abs(&(&velocity - &carried)) < 1e-8);
            let y = product.exp(&x, &d).unwrap();
            assert!((product.norm(&y, &carried) - 0.8).abs() < 1e-12);
        }
    }
}

#[test]
fn distance_log_and_typical_scale_weigh_every_factor() {
    let mut rng = Rng(19);
    for product in [hyperplane(), sphere_by_rotation()] {
        for _ in 0..3 {
            let x = random_point(&product, &mut rng);
            let v = 0.8 * &unit_tangent(&product, &x, &mut rng);
            let y = product.exp(&x, &v).unwrap();
            let back = product.log(&x, &y).unwrap();
            assert!(max_abs(&(&back - &v)) < 1e-10);
            assert!((product.dist(&x, &y).unwrap() - 0.8).abs() < 1e-10);
        }
    }
    let pi2 = std::f64::consts::PI.powi(2);
    let hyperplane_scale = (POSITION_WEIGHT * N as f64 + 7.0 * pi2).sqrt();
    assert!((hyperplane().typical_dist(2 * N) - hyperplane_scale).abs() < 1e-12);
    let rotation_scale = (2.5 * pi2 + 0.4 * 3.0 * pi2).sqrt();
    assert!((sphere_by_rotation().typical_dist(12) - rotation_scale).abs() < 1e-12);
}

const N: usize = 4;
const POSITION_WEIGHT: f64 = 0.3;
const LOWEST: f64 = 1.0;

fn householder() -> Array2<f64> {
    let w = array![1.0, -2.0, 0.5, 1.5];
    let ww = w.dot(&w);
    Array2::from_shape_fn((N, N), |(i, j)| {
        let delta = if i == j { 1.0 } else { 0.0 };
        delta - 2.0 * w[i] * w[j] / ww
    })
}

/// `f(p, u) = (p - c)^T B (p - c) / 2 + u^T A u + kappa ((p - c) . u)^2`
/// on \(\mathbb{R}^4 \times S^3\), with `p` read in `unit`s. The first and
/// last terms vanish at `p = c`, so the minimum is the lowest eigenvalue
/// of `A`, at `u` its eigenvector.
struct Hyperplane {
    a: Array2<f64>,
    b: Array1<f64>,
    c: Array1<f64>,
    kappa: f64,
    unit: f64,
    bounds: Bounds<f64>,
}

impl Hyperplane {
    fn new(unit: f64) -> Self {
        let q = householder();
        let a = q
            .dot(&Array2::from_diag(&array![LOWEST, 2.0, 3.5, 5.0]))
            .dot(&q);
        Self {
            a,
            b: array![1.0, 0.5, 2.0, 1.5],
            c: Self::centre(),
            kappa: 0.7,
            unit,
            bounds: Bounds::new(
                Array1::from_elem(2 * N, -1e6),
                Array1::from_elem(2 * N, 1e6),
                0.0,
            ),
        }
    }

    fn centre() -> Array1<f64> {
        array![0.3, -0.2, 0.5, 0.1]
    }

    fn lowest_mode() -> Array1<f64> {
        householder().column(0).to_owned()
    }

    fn parts(&self, x: ArrayView1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let r = &x.slice(s![..N]) / self.unit - &self.c;
        let u = x.slice(s![N..]).to_owned();
        let overlap = r.dot(&u);
        (r, u, overlap)
    }

    fn value(&self, x: ArrayView1<f64>) -> f64 {
        let (r, u, overlap) = self.parts(x);
        0.5 * (&self.b * &r).dot(&r) + u.dot(&self.a.dot(&u)) + self.kappa * overlap * overlap
    }

    fn gradient(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let (r, u, overlap) = self.parts(x);
        let k = 2.0 * self.kappa * overlap;
        let gp = (&self.b * &r + k * &u) / self.unit;
        let gu = 2.0 * &self.a.dot(&u) + k * &r;
        join(&gp, &gu)
    }

    fn action(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let (r, u, overlap) = self.parts(x);
        let vp = &v.slice(s![..N]) / self.unit;
        let vu = v.slice(s![N..]).to_owned();
        let k = 2.0 * self.kappa;
        let hp = (&self.b * &vp + k * u.dot(&vp) * &u + k * (r.dot(&vu) * &u + overlap * &vu))
            / self.unit;
        let hu =
            2.0 * &self.a.dot(&vu) + k * r.dot(&vu) * &r + k * (u.dot(&vp) * &r + overlap * &vp);
        join(&hp, &hu)
    }
}

impl Objective<f64> for Hyperplane {
    fn dim(&self) -> usize {
        2 * N
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value(x)
    }
}

impl Gradient<f64> for Hyperplane {
    fn dim(&self) -> usize {
        2 * N
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradient(x)
    }
}

impl DifferentiableObjective<f64> for Hyperplane {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.value(x), self.gradient(x))
    }
}

impl HessianVector for Hyperplane {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.action(x, v)
    }
}

fn start(unit: f64) -> Array1<f64> {
    let p = (&Hyperplane::centre() + &array![0.4, -0.3, 0.2, 0.5]) * unit;
    let u: Array1<f64> = array![0.2, 0.9, -0.3, 0.4];
    let norm = u.dot(&u).sqrt();
    join(&p, &(u / norm))
}

fn control(maxiter: usize, gtol: f64) -> Control {
    Control {
        maxiter,
        gtol,
        ..Default::default()
    }
}

fn assert_at_minimum(x: &Array1<f64>, value: f64, what: &str) {
    let (r, u, _) = Hyperplane::new(1.0).parts(x.view());
    assert!((value - LOWEST).abs() < 1e-9, "{what}: f = {value}");
    assert!(
        r.dot(&r).sqrt() < 1e-5,
        "{what}: |p - c| = {}",
        r.dot(&r).sqrt()
    );
    assert!(
        (u.dot(&u).sqrt() - 1.0).abs() < 1e-12,
        "{what}: u left the sphere"
    );
    let overlap = u.dot(&Hyperplane::lowest_mode()).abs();
    assert!(overlap > 1.0 - 1e-9, "{what}: |u . e_min| = {overlap}");
}

fn session(method: Method, maxiter: usize, gtol: f64) -> Solver {
    let mut solver = Solver::new(method, control(maxiter, gtol), 2 * N);
    solver
        .set_product(hyperplane_factors(POSITION_WEIGHT))
        .unwrap();
    solver
}

#[test]
fn riemannian_lbfgs_and_gradient_descent_converge_on_the_hyperplane() {
    let problem = Hyperplane::new(1.0);
    // Gradient descent takes a fixed step below 2 / L for the largest
    // curvature in the product metric, with the energy test as safeguard.
    let descent_control = Control {
        istep: 0.1,
        ..control(5000, 1e-8)
    };
    let mut descent = Solver::new(Method::Steepest, descent_control, 2 * N);
    descent
        .set_product(hyperplane_factors(POSITION_WEIGHT))
        .unwrap();
    descent.set_accept(Accept::Energy);
    for (mut solver, gtol, what) in [
        (session(Method::lbfgs(), 5000, 1e-10), 1e-10, "L-BFGS"),
        (descent, 1e-8, "gradient descent"),
    ] {
        let mut x = start(1.0);
        let report: Report = solver.run(&problem, &mut x).unwrap();
        assert!(
            report.grad_norm < gtol,
            "{what}: |grad| = {}",
            report.grad_norm
        );
        assert_eq!(report.coords, x);
        assert_at_minimum(&x, report.value, what);
    }
}

#[test]
fn session_trust_region_converges_on_the_hyperplane() {
    // The session's Steihaug-Toint step models the Hessian by the projected
    // ambient action, without the Weingarten term of the sphere, so it
    // converges linearly here; the tolerance stays above the level where
    // the decrease of f is lost to rounding.
    let gtol = 1e-7;
    let problem = Hyperplane::new(1.0);
    let mut solver = session(Method::NewtonCg, 500, gtol);
    let mut x = start(1.0);
    let report = solver.run(&problem, &mut x).unwrap();
    assert!(report.grad_norm < gtol);
    assert_at_minimum(&x, report.value, "Newton-CG, difference products");

    let mut solver = session(Method::NewtonCg, 500, gtol);
    let mut x = start(1.0);
    let mut last = None;
    for _ in 0..500 {
        let report = solver.step_hvp(&problem, &mut x).unwrap();
        let done = report.grad_norm < gtol;
        last = Some(report);
        if done {
            break;
        }
    }
    let report = last.unwrap();
    assert!(report.grad_norm < gtol);
    assert_at_minimum(&x, report.value, "Newton-CG, analytic products");
}

fn rtr(unit: f64, position_weight: f64, opts: RtrOptions) -> RtrMin {
    let product = ProductManifold::new(hyperplane_factors(position_weight)).unwrap();
    let problem = Hyperplane::new(unit);
    minimize_rtr(
        &product,
        start(unit),
        |x| problem.value(x.view()),
        |x| problem.gradient(x.view()),
        |x, v| problem.action(x.view(), v.view()),
        opts,
    )
}

#[test]
fn riemannian_trust_region_converges_in_the_product_metric() {
    let out = rtr(
        1.0,
        POSITION_WEIGHT,
        RtrOptions::default().with_grad_tol(1e-11),
    );
    assert!(out.grad_norm <= 1e-11, "|grad| = {}", out.grad_norm);
    assert_at_minimum(&out.x, out.value, "RTR");
    // The Weingarten term makes the model exact to second order, so the
    // inner Newton steps converge superlinearly.
    assert!(out.iterations < 40, "{} iterations", out.iterations);
}

/// L-BFGS iterates with positions read in `unit`s, converted back to the
/// first unit.
fn lbfgs_path(unit: f64, position_weight: f64, steps: usize) -> Vec<Array1<f64>> {
    let problem = Hyperplane::new(unit);
    let mut solver = Solver::new(Method::lbfgs(), control(steps, 0.0), 2 * N);
    solver
        .set_product(hyperplane_factors(position_weight))
        .unwrap();
    let mut x = start(unit);
    let mut path = Vec::new();
    for _ in 0..steps {
        if solver.step(&problem, &mut x).is_err() {
            break;
        }
        let mut natural = x.clone();
        natural.slice_mut(s![..N]).mapv_inplace(|p| p / unit);
        path.push(natural);
    }
    path
}

#[test]
fn iterates_do_not_depend_on_the_length_unit_once_the_weight_carries_it() {
    // Angstrom to bohr: p' = c p, and the weight per squared length
    // becomes w / c^2, so the metric and every iterate are unchanged.
    let bohr = 1.889_726_124_6;
    let reference = lbfgs_path(1.0, POSITION_WEIGHT, 15);
    let rescaled = lbfgs_path(bohr, POSITION_WEIGHT / (bohr * bohr), 15);
    assert_eq!(reference.len(), 15);
    assert_eq!(rescaled.len(), 15);
    for (k, (a, b)) in reference.iter().zip(&rescaled).enumerate() {
        assert!(max_abs(&(a - b)) < 1e-9, "step {k}: {a} vs {b}");
    }
    // Keeping the weight while changing the unit changes the metric.
    let unweighted = lbfgs_path(bohr, POSITION_WEIGHT, 15);
    assert!(
        reference
            .iter()
            .zip(&unweighted)
            .any(|(a, b)| max_abs(&(a - b)) > 1e-4)
    );

    let opts = RtrOptions::default()
        .with_radius_max(2.0)
        .with_grad_tol(1e-10);
    let a = rtr(1.0, POSITION_WEIGHT, opts);
    let b = rtr(bohr, POSITION_WEIGHT / (bohr * bohr), opts);
    assert_eq!(a.iterations, b.iterations);
    assert_eq!(a.accepted, b.accepted);
    let mut natural = b.x.clone();
    natural.slice_mut(s![..N]).mapv_inplace(|p| p / bohr);
    assert!(max_abs(&(&a.x - &natural)) < 1e-9);
}

/// Pulls two three-atom clusters and a 2-vector toward a displaced and
/// deformed target, with a term coupling the first atom to the vector.
struct Pull {
    target: Array1<f64>,
    bounds: Bounds<f64>,
}

impl Pull {
    const LEN: usize = 20;

    fn gradient(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mut g = &x - &self.target;
        g[0] += 0.1 * x[Self::LEN - 1];
        g[Self::LEN - 1] += 0.1 * x[0];
        g
    }
}

impl Objective<f64> for Pull {
    fn dim(&self) -> usize {
        Self::LEN
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let d = &x - &self.target;
        0.5 * d.dot(&d) + 0.1 * x[0] * x[Self::LEN - 1]
    }
}

impl Gradient<f64> for Pull {
    fn dim(&self) -> usize {
        Self::LEN
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.gradient(x)
    }
}

impl DifferentiableObjective<f64> for Pull {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.gradient(x))
    }
}

/// `sum_i m_i x_i` of the cluster in `block`.
fn weighted_centre(x: &Array1<f64>, block: usize, masses: &[f64]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for (i, m) in masses.iter().enumerate() {
        for (d, cd) in c.iter_mut().enumerate() {
            *cd += m * x[9 * block + 3 * i + d];
        }
    }
    c
}

#[test]
fn rigid_factors_keep_their_mass_weighted_centres() {
    let first = [1.0, 16.0, 1.0];
    let second = [12.0, 1.0, 1.0];
    let start = array![
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.5, 0.0, 2.0, 1.0, 0.3,
        0.5, -0.5
    ];
    let mut rng = Rng(23);
    let pull = Pull {
        target: &start + &(0.3 * &rng.vector(Pull::LEN)),
        bounds: Bounds::new(
            Array1::from_elem(Pull::LEN, -1e6),
            Array1::from_elem(Pull::LEN, 1e6),
            0.0,
        ),
    };
    let factors = [
        ProductFactor::new(ManifoldKind::MwRigid, 9).with_weight(2.0),
        ProductFactor::new(ManifoldKind::MwRigid, 9),
        ProductFactor::new(ManifoldKind::Euclidean, 2).with_weight(0.5),
    ];
    // One molecule's masses reused on both clusters, then each cluster's own.
    for (masses, second_masses) in [(first.to_vec(), first), ([first, second].concat(), second)] {
        let mut solver = Solver::new(Method::lbfgs(), control(30, 1e-12), Pull::LEN);
        solver.set_product(factors.clone()).unwrap();
        solver.set_masses(Array1::from(masses));
        let mut x = start.clone();
        solver.run(&pull, &mut x).unwrap();
        for (block, m) in [(0, first), (1, second_masses)] {
            let before = weighted_centre(&start, block, &m);
            let after = weighted_centre(&x, block, &m);
            for d in 0..3 {
                assert!((before[d] - after[d]).abs() < 1e-10, "cluster {block}");
            }
        }
        let moved = weighted_centre(&x, 0, &[1.0; 3]);
        let unmoved = weighted_centre(&start, 0, &[1.0; 3]);
        assert!((0..3).any(|d| (moved[d] - unmoved[d]).abs() > 1e-4));
        assert!(max_abs(&(&x.slice(s![18..]).to_owned() - &start.slice(s![18..]))) > 1e-3);
    }
}

#[test]
fn a_weighted_product_refuses_settings_that_read_raw_coordinates() {
    let problem = Hyperplane::new(1.0);
    let mut solver = session(Method::lbfgs(), 10, 1e-9);
    solver.set_project_rigid(true);
    let mut x = start(1.0);
    let before = x.clone();
    assert!(matches!(
        solver.step(&problem, &mut x),
        Err(Error::ProductChart { .. })
    ));
    assert_eq!(x, before);
}

#[test]
fn power_refuses_a_block_its_factor_cannot_hold() {
    let mut solver = Solver::new(Method::lbfgs(), control(10, 1e-9), 18);
    assert!(!solver.set_power(ManifoldKind::So3, 8, 2));
    assert!(solver.set_power(ManifoldKind::So3, 9, 2));
}
