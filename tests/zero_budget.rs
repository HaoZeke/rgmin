//! A zero iteration budget takes no step, and its report is the objective
//! evaluated at the start: [`Status::Budget`] there, or
//! [`Status::Tolerance`] only when that gradient is below `gtol`.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{Control, FireKind, HessianObjective, HessianVector, Method, Report, Solver, Status};

const GTOL: f64 = 1e-8;

/// `f = 7 + 1/2 (x^2 + 2 y^2 + 3 z^2)`.
struct Bowl {
    bounds: Bounds<f64>,
}

fn bowl() -> Bowl {
    Bowl {
        bounds: Bounds::new(Array1::from_elem(3, -1e6), Array1::from_elem(3, 1e6), 0.0),
    }
}

impl Objective<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.value_and_gradient(x).0
    }
}

impl Gradient<f64> for Bowl {
    fn dim(&self) -> usize {
        3
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.value_and_gradient(x).1
    }
}

impl DifferentiableObjective<f64> for Bowl {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let g = array![x[0], 2.0 * x[1], 3.0 * x[2]];
        (7.0 + 0.5 * x.dot(&g), g)
    }
}

impl HessianObjective for Bowl {
    fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
        Array2::from_diag(&array![1.0, 2.0, 3.0])
    }
}

impl HessianVector for Bowl {
    fn hessian_vector(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        array![v[0], 2.0 * v[1], 3.0 * v[2]]
    }
}

fn moving() -> Array1<f64> {
    array![1.0, -1.0, 0.5]
}

fn stationary() -> Array1<f64> {
    array![0.0, 0.0, 0.0]
}

fn zero_budget() -> Control {
    Control {
        maxiter: 0,
        gtol: GTOL,
        ..Control::default()
    }
}

/// Every first-order method a session or `minimize_method` runs.
fn first_order() -> Vec<Method> {
    vec![
        Method::polak_ribiere(),
        Method::Bfgs,
        Method::lbfgs(),
        Method::Sr1,
        Method::Sr2,
        Method::adam(),
        Method::Steepest,
        Method::pso(),
        Method::Fire { kind: FireKind::V1 },
        Method::Fire { kind: FireKind::V2 },
        Method::Bb,
        Method::QuickMin,
        Method::Ode12r,
        Method::diis(),
        Method::NewtonCg,
        Method::NewtonPcg,
    ]
}

/// The report is `f` and `||g||` at `x0` with no step, `Budget` away
/// from the minimum and `Tolerance` at it.
fn assert_judged_at_start(what: &str, x0: &Array1<f64>, report: &Report) {
    let (f, g) = bowl().value_and_gradient(x0.view());
    let gnorm = g.dot(&g).sqrt();
    assert_eq!(report.steps, 0, "{what}: {report:?}");
    assert_eq!(report.value, f, "{what}: {report:?}");
    assert!(
        (report.grad_norm - gnorm).abs() <= 1e-12 * gnorm.max(1.0),
        "{what}: {report:?}"
    );
    let expected = if gnorm < GTOL {
        Status::Tolerance
    } else {
        Status::Budget
    };
    assert_eq!(report.status, expected, "{what}: {report:?}");
}

#[test]
fn a_zero_budget_session_run_judges_the_evaluated_start() {
    for method in first_order() {
        for x0 in [moving(), stationary()] {
            let mut solver = Solver::new(method.clone(), zero_budget(), 3);
            let mut x = x0.clone();
            let report = solver.run(&bowl(), &mut x).unwrap();
            assert_eq!(x, x0, "{method:?}");
            assert_eq!(solver.steps(), 0, "{method:?}");
            assert_judged_at_start(&format!("run {method:?}"), &x0, &report);
        }
    }
}
