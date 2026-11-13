//! Møller scaled conjugate gradient. The step length comes from a
//! damped model, not a line search.
//!
//! `cargo run --example tutorial_scg`

use ndarray::{ArrayView1, array};
use rgmin::{Conjugacy, Control, Oracle, Restart, ScgParams, minimize_scg};

fn main() {
    let obj = Oracle::unbounded(2, |x: ArrayView1<f64>| {
        let f = (x[0] - 1.0).powi(2) + 4.0 * (x[1] + 2.0).powi(2);
        let g = array![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)];
        (f, g)
    });
    let report = minimize_scg(
        &obj,
        array![0.0, 0.0],
        &Control {
            maxiter: 40,
            gtol: 1e-8,
            ..Control::default()
        },
        &ScgParams::default(),
        Conjugacy::PolakRibiere,
        Restart::Never,
    )
    .unwrap();
    assert!((report.coords[0] - 1.0).abs() < 1e-4, "{}", report.coords);
    assert!((report.coords[1] + 2.0).abs() < 1e-4, "{}", report.coords);
}
