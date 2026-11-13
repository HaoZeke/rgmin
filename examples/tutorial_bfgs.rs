//! Dense BFGS on the Himmelblau function, one of four minima.
//!
//! `cargo run --example tutorial_bfgs`

use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, Oracle, Solver, StopNorm};

fn himmelblau(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let (a, b) = (x[0], x[1]);
    let u = a * a + b - 11.0;
    let v = a + b * b - 7.0;
    let f = u * u + v * v;
    let g = array![4.0 * a * u + 2.0 * v, 2.0 * u + 4.0 * b * v];
    (f, g)
}

fn main() {
    let control = Control {
        maxiter: 80,
        gtol: 1e-8,
        stop_norm: StopNorm::Inf,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Bfgs, control, 2);
    let obj = Oracle::unbounded(2, himmelblau);
    let mut x = array![0.0, 0.0];
    let report = solver.run(&obj, &mut x).unwrap();
    // Four minima. (0, 0) reaches (3.584, -1.848), not (3, 2).
    assert!(report.value < 1e-10, "value {} at {x}", report.value);
}
