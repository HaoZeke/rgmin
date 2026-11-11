// Copyright (c) rgmin developers

//! Opt-in cap measured by the Euclidean length, and the trust radius.
//! Neither is the default. The default still caps the longest atom.

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, LbfgsStep, Method, Solver};

fn spring(x: ArrayView1<f64>, g: &mut Array1<f64>) -> f64 {
    let k = 0.1;
    let mut f = 0.0;
    for i in 0..x.len() {
        g[i] = k * x[i];
        f += 0.5 * k * x[i] * x[i];
    }
    f
}

fn one_step(euclidean: bool) -> Array1<f64> {
    let control = Control {
        maxiter: 1,
        gtol: 1e-18,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control, 6);
    solver
        .set_lbfgs_step(LbfgsStep::Capped { max_atom: 0.2 })
        .unwrap();
    solver.set_lbfgs_euclidean_cap(euclidean).unwrap();
    let mut x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    solver.run_fill(&mut x, |v, buf| spring(v, buf)).unwrap();
    x
}

#[test]
fn the_euclidean_cap_bounds_the_whole_step() {
    let start = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let euclid = one_step(true);
    let atom = one_step(false);
    let en = (&euclid - &start).dot(&(&euclid - &start)).sqrt();
    let an = (&atom - &start).dot(&(&atom - &start)).sqrt();
    assert!((en - 0.2).abs() < 1e-9, "euclidean displacement {en}");
    assert!(
        an > 0.28 && an < 0.29,
        "per-atom cap leaves a longer step, got {an}"
    );
}

#[test]
fn the_trust_radius_still_descends_a_quadratic() {
    let control = Control {
        maxiter: 40,
        gtol: 1e-8,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 8 }, control, 3);
    solver
        .set_lbfgs_step(LbfgsStep::Capped { max_atom: 0.2 })
        .unwrap();
    solver.set_lbfgs_trust(true).unwrap();
    let mut x = array![1.0, 1.0, 1.0];
    let mut energy = 0.0;
    let report = solver
        .run_fill(&mut x, |v, g| {
            energy = spring(v, g);
            energy
        })
        .unwrap();
    assert!(report.grad_norm < 1e-6, "gnorm {}", report.grad_norm);
    assert!(energy < 1e-12, "energy {energy}");
}
