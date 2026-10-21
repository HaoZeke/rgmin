//! Geometry L-BFGS and the Newton / dogleg saddle kick.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::{Control, HessianOracle, Method, NewtonKind, Solver};

fn bowl(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let f = 0.5 * x.dot(&x);
    (f, x.to_owned())
}

fn well(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let y = x[1];
    let f = (y * y - 1.0).powi(2) + 0.5 * x[0] * x[0];
    let g = array![x[0], 4.0 * y * (y * y - 1.0)];
    (f, g)
}

fn well_h(x: ArrayView1<f64>) -> Array2<f64> {
    let y = x[1];
    Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 12.0 * y * y - 4.0]).unwrap()
}

fn control(gtol: f64) -> Control {
    Control {
        maxiter: 80,
        gtol,
        istep: 0.2,
        maxmove: None,
    }
}

#[test]
fn geometry_lbfgs_reaches_the_bowl_with_a_per_atom_cap() {
    let obj = rgmin::Oracle::unbounded(3, bowl);
    let mut solver = Solver::new(Method::lbfgs(), control(1e-8), 3);
    solver.set_atom_maxmove(0.2);
    let mut x = array![1.2, -0.4, 0.7];
    let mut last = f64::MAX;
    for _ in 0..80 {
        let rep = solver.step(&obj, &mut x).expect("step");
        last = rep.grad_norm;
        if last <= 1e-8 {
            break;
        }
    }
    assert!(last <= 1e-8, "geometry L-BFGS |g|={last} x={x:?}");
    assert!(x.iter().all(|v| v.abs() < 1e-6), "x={x:?}");
}

#[test]
fn dogleg_and_newton_leave_a_saddle() {
    let obj = HessianOracle::unbounded(2, well, well_h);
    let gtol = 1e-8;
    let methods = [
        ("dogleg", Method::Dogleg),
        (
            "newton",
            Method::Newton {
                kind: NewtonKind::Shifted,
            },
        ),
    ];
    for (name, method) in methods {
        let mut solver = Solver::new(method, control(gtol), 2);
        solver.set_atom_maxmove(0.25);
        let mut x = array![0.0, 0.0];
        let first = solver.step_hess(&obj, &mut x).expect("kick");
        assert!(
            first.grad_norm > gtol,
            "{name} stayed on the saddle |g|={} x={x:?}",
            first.grad_norm
        );
        assert!(x[1].abs() > 0.2, "{name} kick was {x:?}");
        let mut last = first.grad_norm;
        for _ in 0..60 {
            if last <= gtol {
                break;
            }
            let rep = solver.step_hess(&obj, &mut x).expect("step");
            last = rep.grad_norm;
        }
        assert!(last <= gtol, "{name} |g|={last} x={x:?}");
        assert!(
            (x[1].abs() - 1.0).abs() < 1e-4 && x[0].abs() < 1e-4,
            "{name} ended at {x:?}"
        );
    }
}

#[test]
fn uncapped_lbfgs_still_takes_the_unit_newton_step() {
    let obj = rgmin::Oracle::unbounded(2, bowl);
    let mut solver = Solver::new(Method::lbfgs(), control(1e-12), 2);
    let mut x = array![0.4, -0.2];
    let rep = solver.step(&obj, &mut x).expect("step");
    assert!(
        rep.grad_norm < 1e-8,
        "uncapped L-BFGS |g|={} x={x:?}",
        rep.grad_norm
    );
}
