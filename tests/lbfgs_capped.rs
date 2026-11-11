// Copyright (c) rgmin developers

//! Scaled L-BFGS keeps its pairs when the step is shortened.
//!
//! The geometry step selected by `set_atom_maxmove` forgets the history
//! whenever the quasi-Newton step is longer than the cap. The scaled
//! step shortens that direction and stores the pair. The default step
//! is the eOn cap, which clears that history.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, FireKind, LbfgsStep, Method, Oracle, Solver};

fn quadratic(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    (0.5 * x.dot(&x), x.to_owned())
}

fn session(scaled: bool) -> Solver {
    let control = Control {
        maxiter: 20,
        gtol: 1e-12,
        istep: 1.0,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control, 3);
    if scaled {
        solver
            .set_lbfgs_step(LbfgsStep::Scaled { max_atom: 0.2 })
            .unwrap();
    } else {
        solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
        solver.set_atom_maxmove(0.2);
    }
    solver
}

/// Four steps from `(2, 0, 0)` on `f = 1/2 ||x||^2`. The Newton step
/// stays longer than the cap, so the geometry step retains one pair
/// and the scaled step retains the window.
#[test]
fn scaled_keeps_the_history_the_geometry_step_drops() {
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
        "scaled history {}",
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

/// The first scaled session step moves the atom by the cap, and each
/// later step adds one oracle call.
#[test]
fn a_scaled_session_step_costs_one_evaluation() {
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
fn a_non_positive_cap_restores_the_eon_default() {
    let mut solver = session(true);
    solver
        .set_lbfgs_step(LbfgsStep::Capped { max_atom: 0.0 })
        .unwrap();
    assert_eq!(solver.lbfgs_step(), LbfgsStep::default());
}

/// A session that never selects a step rule uses the eOn cap.
/// The first step spends a curvature probe, so the oracle runs three
/// times, and the atom moves by `EON_MAX_MOVE`.
#[test]
fn the_default_session_probes_then_moves_by_the_eon_cap() {
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let obj = Oracle::unbounded(3, |x: ArrayView1<f64>| {
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        quadratic(x)
    });
    let control = Control {
        gtol: 1e-12,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control, 3);
    assert_eq!(solver.lbfgs_step(), LbfgsStep::default());
    let mut x = array![1.0, 0.0, 0.0];
    solver.step(&obj, &mut x).unwrap();
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 3);
    assert!((x[0] - (1.0 - rgmin::EON_MAX_MOVE)).abs() <= 1e-12, "{x}");
    solver.step(&obj, &mut x).unwrap();
    let force_step = 0.8 - x[0];
    assert!((force_step - rgmin::EON_MAX_MOVE).abs() <= 1e-12, "{x}");
    assert_eq!(
        solver.pair_count(),
        1,
        "a step of length 0.2 from 0.8 clears the quasi-Newton pair"
    );
}

/// The buffer-filling driver takes the same 66 evaluations as eOn on
/// Rosenbrock 2 with memory 20.
#[test]
fn run_fill_matches_the_eon_rosenbrock2_count() {
    let mut calls = 0usize;
    let mut x = array![-1.2, 1.0];
    let control = Control {
        maxiter: 20_000,
        gtol: 1e-5,
        istep: 1.0,
        stop_norm: rgmin::StopNorm::Inf,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 20 }, control, 2);
    solver
        .set_lbfgs_step(LbfgsStep::Capped { max_atom: 0.2 })
        .unwrap();
    let rep = solver
        .run_fill(&mut x, |x, g| {
            calls += 1;
            g.fill(0.0);
            let mut f = 0.0;
            let xs = x.as_slice().unwrap();
            for i in 0..xs.len() - 1 {
                let a = 1.0 - xs[i];
                let b = xs[i + 1] - xs[i] * xs[i];
                f += a * a + 100.0 * b * b;
                g[i] += -2.0 * a - 400.0 * xs[i] * b;
                g[i + 1] += 200.0 * b;
            }
            f
        })
        .unwrap();
    assert!(rep.grad_norm < 1e-5, "{}", rep.grad_norm);
    assert_eq!(calls, 66, "evals {calls}");
}
