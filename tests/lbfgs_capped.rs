// Copyright (c) rgmin developers

//! Capped L-BFGS keeps its pairs when the step is shortened.
//!
//! The geometry step selected by `set_atom_maxmove` forgets the history
//! whenever the quasi-Newton step is longer than the cap. The capped
//! step shortens that direction and stores the pair.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, FireKind, LbfgsStep, Method, Oracle, Solver};

fn quadratic(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (0.5 * x.dot(&x), x.to_owned())
}

fn session(capped: bool) -> Solver {
    let control = Control {
        maxiter: 20,
        gtol: 1e-12,
        istep: 1.0,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control, 3);
    if capped {
        solver
            .set_lbfgs_step(LbfgsStep::Capped { max_atom: 0.2 })
            .unwrap();
    } else {
        solver.set_atom_maxmove(0.2);
    }
    solver
}

/// Four steps from `(2, 0, 0)` on `f = 1/2 ||x||^2`. The Newton step
/// stays longer than the cap, so the geometry step retains one pair
/// and the capped step retains the window.
#[test]
fn capped_keeps_the_history_the_geometry_step_drops() {
    let obj = Oracle::unbounded(3, quadratic);
    let mut x_cap = array![2.0, 0.0, 0.0];
    let mut x_geo = x_cap.clone();
    let mut capped = session(true);
    let mut geometry = session(false);
    for _ in 0..4 {
        capped.step(&obj, &mut x_cap).unwrap();
        geometry.step(&obj, &mut x_geo).unwrap();
    }
    assert!(
        capped.pair_count() >= 3,
        "capped history {}",
        capped.pair_count()
    );
    assert!(
        geometry.pair_count() <= 1,
        "geometry history {}",
        geometry.pair_count()
    );
    let motion = (2.0 - x_cap[0]).abs();
    assert!(
        (motion - 0.8).abs() <= 1e-12,
        "four caps of 0.2, moved {motion}"
    );
    assert!(x_cap[1].abs() <= 1e-12 && x_cap[2].abs() <= 1e-12);
}

/// The first capped session step moves the atom by the cap, and each
/// later step adds one oracle call.
#[test]
fn a_capped_session_step_costs_one_evaluation() {
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let obj = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        quadratic(x)
    });
    let mut solver = session(true);
    let mut x = array![1.0, 0.0, 0.0];
    solver.step(&obj, &mut x).unwrap();
    assert!((x[0] - 0.8).abs() <= 1e-12, "x {x}");
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);
    solver.step(&obj, &mut x).unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
    assert!((x[0] - 0.6).abs() <= 1e-12, "x {x}");
}

#[test]
fn the_capped_step_is_refused_off_an_lbfgs_session() {
    let mut solver = Solver::new(Method::Fire { kind: FireKind::V1 }, Control::default(), 2);
    let err = solver
        .set_lbfgs_step(LbfgsStep::Capped { max_atom: 0.2 })
        .unwrap_err();
    assert!(err.to_string().contains("L-BFGS"), "{err}");
}

#[test]
fn a_non_positive_cap_restores_the_wolfe_rule() {
    let mut solver = session(true);
    solver
        .set_lbfgs_step(LbfgsStep::Capped { max_atom: 0.0 })
        .unwrap();
    assert_eq!(solver.lbfgs_step(), LbfgsStep::Wolfe);
}
