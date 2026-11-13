//! FIRE 2.0 with the Guénolé parameters on a quadratic bowl.
//!
//! `cargo run --example tutorial_fire`

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, FireKind, FirePreset, Method, Oracle, Solver};

fn main() {
    let control = Control {
        maxiter: 500,
        gtol: 1e-6,
        istep: 0.1,
        maxmove: Some(0.2),
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Fire { kind: FireKind::V2 }, control, 2);
    solver.set_fire_preset(FirePreset::Guenole2020).unwrap();
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        (x.dot(&x), Array1::from_iter(x.iter().map(|v| 2.0 * v)))
    });
    let mut x = array![0.8, -0.3];
    let report = solver.run(&obj, &mut x).unwrap();
    assert!(report.value < 1e-8, "value {}", report.value);
    assert!(x.iter().all(|v| v.abs() < 1e-4), "{x}");
}
