//! On a curved kind, every point a session evaluates or accepts lies on
//! the set, and every accepted move is within the displacement cap.

use std::sync::Mutex;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{Accept, Control, FireKind, HessianObjective, ManifoldKind, Method, Solver};

const ON_SPHERE: f64 = 1e-10;
const STEPS: usize = 12;
const CAP: f64 = 0.1;
/// A norm recomputed in another summation order differs by a few ulps.
const CAP_ROUNDING: f64 = 8.0 * f64::EPSILON;

fn control() -> Control {
    Control {
        maxiter: STEPS,
        gtol: 1e-12,
        istep: 1.0,
        maxmove: Some(CAP),
    }
}

fn distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let d = a - b;
    d.dot(&d).sqrt()
}

/// f(x) = -x_3 + c |x|^2 / 2. The second term is constant on the unit
/// sphere, and c = 1 gives dogleg the Hessian I.
struct Tilt {
    curvature: f64,
    bounds: Bounds<f64>,
    offsets: Mutex<Vec<f64>>,
    energies: Mutex<usize>,
}

impl Tilt {
    fn new(curvature: f64) -> Self {
        Self {
            curvature,
            bounds: Bounds::new(Array1::from_elem(3, -1e6), Array1::from_elem(3, 1e6), 0.0),
            offsets: Mutex::new(Vec::new()),
            energies: Mutex::new(0),
        }
    }

    fn note(&self, x: ArrayView1<f64>) {
        self.offsets
            .lock()
            .unwrap()
            .push((x.dot(&x).sqrt() - 1.0).abs());
    }

    fn energy(&self, x: ArrayView1<f64>) -> f64 {
        self.note(x);
        *self.energies.lock().unwrap() += 1;
        -x[2] + 0.5 * self.curvature * x.dot(&x)
    }
}

impl Objective<f64> for Tilt {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.energy(x)
    }
}

impl Gradient<f64> for Tilt {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.note(x);
        array![0.0, 0.0, -1.0] + self.curvature * &x
    }
}

impl DifferentiableObjective<f64> for Tilt {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let value = self.energy(x);
        (value, array![0.0, 0.0, -1.0] + self.curvature * &x)
    }
}

impl HessianObjective for Tilt {
    fn hessian(&self, x: ArrayView1<f64>) -> Array2<f64> {
        self.note(x);
        Array2::eye(3) * self.curvature
    }
}

struct Walk {
    evaluated: f64,
    accepted: f64,
    largest_move: f64,
    energies: usize,
}

/// Twelve steps from (1, 0, 0) with the Euclidean cap.
fn walk(method: Method, accept: Accept, curvature: f64) -> Walk {
    let objective = Tilt::new(curvature);
    let second_order = matches!(method, Method::Dogleg);
    let mut solver = Solver::new(method, control(), 3);
    solver.set_manifold(ManifoldKind::Sphere);
    solver.set_accept(accept);
    let mut x = array![1.0, 0.0, 0.0];
    let mut accepted = 0.0_f64;
    let mut largest_move = 0.0_f64;
    for _ in 0..STEPS {
        let before = x.clone();
        if second_order {
            solver.step_hess(&objective, &mut x).unwrap();
        } else {
            solver.step(&objective, &mut x).unwrap();
        }
        accepted = accepted.max((x.dot(&x).sqrt() - 1.0).abs());
        largest_move = largest_move.max(distance(&x, &before));
    }
    assert!(x[2] > 0.5, "the walk did not climb toward the pole: {x}");
    let offsets = objective.offsets.into_inner().unwrap();
    Walk {
        evaluated: offsets.iter().copied().fold(0.0, f64::max),
        accepted,
        largest_move,
        energies: objective.energies.into_inner().unwrap(),
    }
}

fn assert_on_sphere(walk: &Walk) {
    assert!(
        walk.evaluated <= ON_SPHERE,
        "an evaluated point is {:.3e} off the sphere",
        walk.evaluated
    );
    assert!(
        walk.accepted <= ON_SPHERE,
        "an accepted point is {:.3e} off the sphere",
        walk.accepted
    );
    assert!(
        walk.largest_move <= CAP * (1.0 + CAP_ROUNDING),
        "an accepted step moved {:.6e}, over the cap {CAP}",
        walk.largest_move
    );
}

#[test]
fn steepest_with_the_energy_test_stays_on_the_sphere() {
    let walk = walk(Method::Steepest, Accept::Energy, 0.0);
    assert_on_sphere(&walk);
    assert!(walk.energies <= STEPS + 1, "{} energies", walk.energies);
}

#[test]
fn steepest_with_one_trial_stays_on_the_sphere() {
    let walk = walk(Method::Steepest, Accept::None, 0.0);
    assert_on_sphere(&walk);
    assert!(walk.energies <= STEPS + 1, "{} energies", walk.energies);
}

#[test]
fn fire_v1_stays_on_the_sphere() {
    let walk = walk(Method::Fire { kind: FireKind::V1 }, Accept::None, 0.0);
    assert_on_sphere(&walk);
    assert!(walk.energies <= STEPS + 1, "{} energies", walk.energies);
}

#[test]
fn fire_v2_stays_on_the_sphere() {
    let walk = walk(Method::Fire { kind: FireKind::V2 }, Accept::None, 0.0);
    assert_on_sphere(&walk);
}

#[test]
fn dogleg_stays_on_the_sphere() {
    let walk = walk(Method::Dogleg, Accept::None, 1.0);
    assert_on_sphere(&walk);
}

/// f(x) = -x_3 + sum_i d_i x_i^2 / 2 with distinct d_i, so the tangent
/// model differs from the Hessian at every point of the walk.
struct Bowl {
    bounds: Bounds<f64>,
    weights: Array1<f64>,
    energies: Mutex<usize>,
}

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        *self.energies.lock().unwrap() += 1;
        -x[2] + 0.5 * (&self.weights * &x).dot(&x)
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![0.0, 0.0, -1.0] + &self.weights * &x
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

impl HessianObjective for Bowl {
    fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
        Array2::from_diag(&self.weights)
    }
}

fn dogleg_end<O: HessianObjective>(objective: &O, start: Array1<f64>) -> Array1<f64> {
    let mut solver = Solver::new(Method::Dogleg, control(), 3);
    solver.set_manifold(ManifoldKind::Sphere);
    solver.set_accept(Accept::None);
    let mut x = start;
    for _ in 0..STEPS {
        solver.step_hess(objective, &mut x).unwrap();
    }
    x
}

fn assert_end(got: &Array1<f64>, want: [f64; 3]) {
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-12, "endpoint {got}, want {want:?}");
    }
}

#[test]
fn dogleg_on_the_sphere_keeps_its_endpoint_and_calls() {
    let tilt = Tilt::new(1.0);
    let end = dogleg_end(&tilt, array![1.0, 0.0, 0.0]);
    assert_end(&end, [0.366060826241496, 0.0, 0.930590925966610]);
    assert_eq!(tilt.energies.into_inner().unwrap(), STEPS + 1);
    let bowl = Bowl {
        bounds: Bounds::new(Array1::from_elem(3, -1e6), Array1::from_elem(3, 1e6), 0.0),
        weights: array![1.0, 2.0, 4.0],
        energies: Mutex::new(0),
    };
    let end = dogleg_end(&bowl, array![0.48, 0.6, 0.64]);
    assert_end(
        &end,
        [0.942775681942755, 0.005214117204762, 0.333386902140963],
    );
    assert_eq!(bowl.energies.into_inner().unwrap(), STEPS + 1);
}

#[test]
fn a_sphere_dogleg_after_a_positive_dogleg_keeps_its_endpoint() {
    let bowl = Bowl {
        bounds: Bounds::new(Array1::from_elem(3, 1e-6), Array1::from_elem(3, 1e6), 0.0),
        weights: array![1.0, 2.0, 4.0],
        energies: Mutex::new(0),
    };
    let mut solver = Solver::new(Method::Dogleg, control(), 3);
    solver.set_manifold(ManifoldKind::positive(3));
    solver.set_accept(Accept::None);
    let mut x = array![0.48, 0.6, 0.64];
    for _ in 0..STEPS {
        solver.step_hess(&bowl, &mut x).unwrap();
    }
    assert!(x.iter().all(|v| *v > 0.0), "left the orthant: {x}");
    solver.set_manifold(ManifoldKind::Sphere);
    let tilt = Tilt::new(1.0);
    let mut x = array![1.0, 0.0, 0.0];
    for _ in 0..STEPS {
        solver.step_hess(&tilt, &mut x).unwrap();
    }
    assert_end(&x, [0.366060826241496, 0.0, 0.930590925966610]);
    assert_eq!(tilt.energies.into_inner().unwrap(), STEPS + 1);
}

/// f(x) = -(x_1 + x_2 + x_3) on the positive orthant. Descent grows x,
/// where the retraction x exp(v / x) moves farther than the step v.
struct Climb {
    bounds: Bounds<f64>,
    energies: Mutex<usize>,
}

impl Objective<f64> for Climb {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        *self.energies.lock().unwrap() += 1;
        -x.sum()
    }
}

impl Gradient<f64> for Climb {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_elem(3, -1.0)
    }
}

impl DifferentiableObjective<f64> for Climb {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

#[test]
fn an_expanding_retraction_is_shrunk_to_the_cap() {
    let objective = Climb {
        bounds: Bounds::new(Array1::from_elem(3, 1e-6), Array1::from_elem(3, 1e6), 0.0),
        energies: Mutex::new(0),
    };
    let mut solver = Solver::new(Method::Steepest, control(), 3);
    solver.set_manifold(ManifoldKind::positive(3));
    solver.set_accept(Accept::None);
    let mut x = array![1.0, 1.0, 1.0];
    let mut largest_move = 0.0_f64;
    for _ in 0..STEPS {
        let before = x.clone();
        solver.step(&objective, &mut x).unwrap();
        assert!(x.iter().all(|v| *v > 0.0), "left the orthant: {x}");
        largest_move = largest_move.max(distance(&x, &before));
    }
    assert!(x.sum() > 3.0, "the walk did not descend: {x}");
    assert!(
        largest_move <= CAP * (1.0 + CAP_ROUNDING),
        "an accepted step moved {largest_move:.6e}, over the cap {CAP}"
    );
    let energies = objective.energies.into_inner().unwrap();
    assert!(energies <= STEPS + 1, "{energies} energies");
}
