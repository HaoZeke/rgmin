//! An accepted L-BFGS step of about 1e-10 that leaves the value and the
//! gradient norm bitwise unchanged, and then keeps doing it.
//!
//! A caller with `maxiter == 0` keeps taking steps while the report is
//! still iterating. On e7e66b60 that report stays `Iterating`, so these
//! tests do not return. The stop waits `max(n, 2)` accepted steps: one
//! flat step is the whole story of a 1-D overshoot, and a wider problem
//! waits out its own dimension.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, EON_MAX_MOVE, Lbfgs, LbfgsStep, LineSearch, Method, Solver, Status};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const VALUE: f64 = 759_036.0;
const GRAD: f64 = 2.14983;
/// A trial farther than this from the current point is uphill, so a unit
/// Wolfe trial is refused and the search accepts inside the ball.
const FLAT: f64 = 2e-10;

struct ConstantForce {
    dim: usize,
    anchor: Mutex<Array1<f64>>,
    bounds: Bounds<f64>,
}

impl ConstantForce {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            anchor: Mutex::new(Array1::zeros(dim)),
            bounds: Bounds::new(
                Array1::from_elem(dim, -1e6),
                Array1::from_elem(dim, 1e6),
                0.0,
            ),
        }
    }

    fn set_anchor(&self, x: &Array1<f64>) {
        *self.anchor.lock().unwrap() = x.clone();
    }
}

impl Objective<f64> for ConstantForce {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let anchor = self.anchor.lock().unwrap();
        let r = (&x - &*anchor).mapv(|v| v * v).sum().sqrt();
        if r > FLAT { VALUE + (r - FLAT) } else { VALUE }
    }
}

impl Gradient<f64> for ConstantForce {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let anchor = self.anchor.lock().unwrap();
        let d = &x - &*anchor;
        let r = d.mapv(|v| v * v).sum().sqrt();
        if r > FLAT {
            return &d * (100.0 / r);
        }
        let mut g = Array1::zeros(self.dim);
        g[0] = -GRAD;
        g
    }
}

impl DifferentiableObjective<f64> for ConstantForce {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

/// Two-dimensional Wolfe objective. Off the anchor the gradient is
/// perpendicular to the step, so strong curvature holds and the norm
/// does not change. That is the H2CO retry: the line search, not the cap.
struct WolfePlane {
    anchor: Mutex<Array1<f64>>,
    bounds: Bounds<f64>,
    calls: AtomicUsize,
}

impl WolfePlane {
    fn new() -> Self {
        Self {
            anchor: Mutex::new(array![0.0, 0.0]),
            bounds: Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0),
            calls: AtomicUsize::new(0),
        }
    }

    fn set_anchor(&self, x: &Array1<f64>) {
        *self.anchor.lock().unwrap() = x.clone();
    }
}

impl Objective<f64> for WolfePlane {
    fn dim(&self) -> usize {
        2
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        let anchor = self.anchor.lock().unwrap();
        let d = &x - &*anchor;
        let r = d.dot(&d).sqrt();
        if r > FLAT { VALUE + (r - FLAT) } else { VALUE }
    }
}

impl Gradient<f64> for WolfePlane {
    fn dim(&self) -> usize {
        2
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let anchor = self.anchor.lock().unwrap();
        let dx = x[0] - anchor[0];
        let dy = x[1] - anchor[1];
        let r = (dx * dx + dy * dy).sqrt();
        if r > FLAT {
            return array![100.0 * dx / r, 100.0 * dy / r];
        }
        if r == 0.0 {
            return array![-GRAD, 0.0];
        }
        array![-GRAD * dy / r, GRAD * dx / r]
    }
}

impl DifferentiableObjective<f64> for WolfePlane {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.eval(x), self.grad(x))
    }
}

fn session(dim: usize, maxiter: usize, wolfe: bool) -> Solver {
    let mut solver = Solver::new(
        Method::lbfgs(),
        Control {
            maxiter,
            gtol: 1e-2,
            istep: 1.0,
            // A negative cap is the hyperparameter session: no Euclidean
            // limit. The default step stays the capped eOn move unless
            // this test opts into Wolfe.
            maxmove: Some(-1.0),
            ..Control::default()
        },
        dim,
    );
    if wolfe {
        solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
        solver.set_atom_maxmove(0.0);
        solver.set_linesearch(LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 80,
        });
    } else {
        solver.set_lbfgs_capped(1e-10).unwrap();
        solver.set_atom_maxmove(1e-10);
    }
    solver
}

fn assert_about_1e_10(step: f64) {
    assert!(
        (step - 1e-10).abs() < 5e-10,
        "accepted step {step} is not about 1e-10"
    );
}

fn assert_frozen(value: f64, grad_norm: f64) {
    assert_eq!(value.to_bits(), VALUE.to_bits());
    assert!(
        (grad_norm - GRAD).abs() <= GRAD * f64::EPSILON.sqrt(),
        "grad {grad_norm}"
    );
    assert!(grad_norm >= 1e-2, "grad {grad_norm} is above the tolerance");
}

fn step_length(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    (after - before).mapv(|v| v * v).sum().sqrt()
}

fn drive_capped(dim: usize, maxiter: usize) -> (Status, usize, f64, f64) {
    let objective = ConstantForce::new(dim);
    let mut solver = session(dim, maxiter, false);
    let mut x = Array1::zeros(dim);
    let mut accepted = 0usize;
    loop {
        let before = x.clone();
        let report = solver.step(&objective, &mut x).unwrap();
        accepted += 1;
        assert_about_1e_10(step_length(&before, &x));
        assert_frozen(report.value, report.grad_norm);
        objective.set_anchor(&x);
        if maxiter > 0 {
            assert_eq!(report.status, Status::Iterating, "{report:?}");
            if accepted == maxiter {
                return (report.status, accepted, report.value, report.grad_norm);
            }
        } else if report.status != Status::Iterating {
            return (report.status, accepted, report.value, report.grad_norm);
        }
    }
}

fn drive_wolfe() -> (Status, usize) {
    let objective = WolfePlane::new();
    let mut solver = session(2, 0, true);
    let mut x = array![0.0, 0.0];
    let mut accepted = 0usize;
    loop {
        let before = x.clone();
        let report = solver.step(&objective, &mut x).unwrap();
        accepted += 1;
        assert_about_1e_10(step_length(&before, &x));
        assert_frozen(report.value, report.grad_norm);
        objective.set_anchor(&x);
        if report.status != Status::Iterating {
            assert!(
                objective.calls.load(Ordering::Relaxed) < 2_000,
                "evaluations {}",
                objective.calls.load(Ordering::Relaxed)
            );
            return (report.status, accepted);
        }
    }
}

#[test]
fn the_default_step_stays_the_capped_eon_move() {
    assert_eq!(
        Lbfgs::default().step,
        LbfgsStep::Capped {
            max_atom: EON_MAX_MOVE
        }
    );
    assert_eq!(
        Solver::new(Method::lbfgs(), Control::default(), 2).lbfgs_step(),
        LbfgsStep::Capped {
            max_atom: EON_MAX_MOVE
        }
    );
}

#[test]
fn one_dimensional_unlimited_fit_stops_after_two_repeated_steps() {
    let (status, accepted, _, grad) = drive_capped(1, 0);
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 1usize.max(2));
    assert!(grad >= 1e-2);
}

#[test]
fn four_dimensional_unlimited_fit_stops_after_its_period() {
    let (status, accepted, _, grad) = drive_capped(4, 0);
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 4usize.max(2));
    assert!(grad >= 1e-2);
}

#[test]
fn wolfe_plane_stops_on_the_same_period() {
    let (status, accepted) = drive_wolfe();
    assert_eq!(status, Status::StepTolerance);
    assert_eq!(accepted, 2);
}

#[test]
fn a_positive_budget_walks_past_the_one_dimensional_period() {
    let (status, accepted, _, _) = drive_capped(1, 4);
    assert_eq!(status, Status::Iterating);
    assert_eq!(accepted, 4);
}
