//! Minimize the 2-D Rosenbrock banana with the default L-BFGS step.
//!
//! `cargo run --example tutorial_quickstart`

use ndarray::array;
use rgmin::Lbfgs;

fn main() {
    let mut opt = Lbfgs::default();
    let x0 = array![-1.2, 1.0];
    let (_f, x, _evals) = opt.minimize(x0.view(), 200, |x| {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        Some((
            a * a + 100.0 * b * b,
            array![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
        ))
    });
    assert!(x[0] > 0.9);
}
