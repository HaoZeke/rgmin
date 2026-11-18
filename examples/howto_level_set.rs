//! Keep a session on the level set `value(x) = target`.
//!
//! `cargo run --example howto_level_set`

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, LevelSetConstraint, Method, Oracle, Solver};

fn radius_squared(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (x.dot(&x), x.mapv(|v| 2.0 * v))
}

fn main() {
    let obj = Oracle::unbounded(2, radius_squared);
    let target = 1.0;
    let control = Control::default().with_maxiter(3).with_gtol(0.0);
    let mut solver = Solver::new(Method::lbfgs(), control, 2);
    solver
        .set_level_set(LevelSetConstraint::new(target))
        .unwrap();

    // The start is off the set. Each accepted point is back on it.
    let mut x = array![2.0, 0.0];
    for _ in 0..3 {
        solver.step(&obj, &mut x).unwrap();
        let (value, _) = obj.value_and_gradient(x.view());
        assert!((value - target).abs() < 1e-6, "value {value} at {x}");
    }

    solver.clear_level_set();
    assert!(solver.level_set().is_none());
}
