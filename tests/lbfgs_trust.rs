//! Opt-in Burdakov ray truncation for session L-BFGS.
//!
//! `set_lbfgs_trust` stays the eOn Newton-model radius. The radius here
//! is off by default, and the capped step stays the default until it is on.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver};

fn control() -> Control {
    Control::default()
        .with_maxiter(20)
        .with_gtol(1e-8)
        .with_istep(1.0)
}

fn bowl(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (0.5 * x.dot(&x), x.to_owned())
}

#[test]
fn the_radius_is_off_until_a_positive_value_is_stored() {
    let mut solver = Solver::new(Method::lbfgs(), control(), 2);
    assert_eq!(solver.lbfgs_trust_radius(), None);
    solver.set_lbfgs_trust_radius(0.5);
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.5));
    solver.set_lbfgs_trust_radius(0.0);
    assert_eq!(solver.lbfgs_trust_radius(), None);
    solver.set_lbfgs_trust_radius(f64::NAN);
    assert_eq!(solver.lbfgs_trust_radius(), None);
}

#[test]
fn a_step_under_the_radius_moves_and_keeps_a_positive_radius() {
    let obj = Oracle::unbounded(2, bowl);
    let mut solver = Solver::new(Method::lbfgs(), control(), 2);
    solver.set_lbfgs_trust_radius(0.5);
    solver.set_atom_maxmove(10.0);
    let mut x = array![1.0, -0.5];
    let start = x.clone();
    let report = solver.step(&obj, &mut x).unwrap();
    assert_ne!(x, start);
    assert!(report.value < 0.5 * start.dot(&start));
    let radius = solver.lbfgs_trust_radius().unwrap();
    assert!(radius.is_finite() && radius > 0.0, "{radius}");
    solver.forget();
    assert_eq!(solver.lbfgs_trust_radius(), Some(0.5));
}

/// The default Euclidean step is the eOn cap, and that path returns
/// before the trust step. With the radius on, a zero gradient at
/// `gtol = 0` is no descent, and a slope that overflows is a non-finite
/// direction. Both leave `x` where it was.
#[test]
fn the_eon_fast_path_does_not_bypass_the_radius() {
    let flat_control = control().with_gtol(0.0);
    let flat = Oracle::unbounded(2, |x: ArrayView1<f64>| (1.0, Array1::zeros(x.len())));
    let mut solver = Solver::new(Method::lbfgs(), flat_control, 2);
    solver.set_lbfgs_trust_radius(0.5);
    let mut x = array![3.0, -1.0];
    let err = solver.step(&flat, &mut x).unwrap_err();
    assert!(err.to_string().contains("no descent direction"), "{err}");
    assert_eq!(x, array![3.0, -1.0]);

    let steep = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        (1.0, Array1::from_elem(x.len(), 1e200))
    });
    let mut solver = Solver::new(Method::lbfgs(), control(), 2);
    solver.set_lbfgs_trust_radius(0.5);
    let mut x = array![3.0, -1.0];
    let err = solver.step(&steep, &mut x).unwrap_err();
    assert!(
        err.to_string().contains("non-finite search direction"),
        "{err}"
    );
    assert_eq!(x, array![3.0, -1.0]);
}
