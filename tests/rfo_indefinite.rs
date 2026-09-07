use ndarray::{array, ArrayView1};
use rgmin::{minimize_newton, Control, HessianOracle, NewtonKind};

#[test]
fn rfo_minimization_selects_the_lowest_augmented_mode() {
    let objective = HessianOracle::unbounded(
        1,
        |x: ArrayView1<f64>| {
            let q = x[0];
            (
                0.25 * q.powi(4) - 0.5 * q * q + 0.1 * q,
                array![q.powi(3) - q + 0.1],
            )
        },
        |x: ArrayView1<f64>| array![[3.0 * x[0] * x[0] - 1.0]],
    );
    let report = minimize_newton(
        &objective,
        array![0.0],
        &Control {
            maxiter: 1,
            gtol: 1e-12,
            istep: 1.0,
            maxmove: Some(0.2),
        },
        NewtonKind::Rfo,
    )
    .unwrap();

    // The lowest eigenvalue of [[-1, 0.1], [0.1, 0]] is negative.
    // Its RFO direction points downhill and reaches the movement limit.
    assert!(
        (report.coords[0] + 0.2).abs() < 1e-12,
        "RFO step {} does not follow the lowest augmented mode",
        report.coords[0]
    );
    assert!((report.value + 0.0396).abs() < 1e-12);
}
