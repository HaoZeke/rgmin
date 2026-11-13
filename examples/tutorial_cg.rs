//! Line-search-free conjugate gradients, the eOn client step.
//!
//! `cargo run --example tutorial_cg`

use rgmin::{EonCg, minimize_eon_cg};

fn main() {
    let mut x = vec![1.0, -2.0, 0.5];
    let opts = EonCg {
        gtol: 1e-8,
        max_iter: 200,
        ..EonCg::default()
    };
    let report = minimize_eon_cg(
        &mut x,
        |xs, gs| {
            let mut f = 0.0;
            for (x, g) in xs.iter().zip(gs.iter_mut()) {
                f += x * x;
                *g = 2.0 * x;
            }
            f
        },
        &opts,
    );
    assert!(report.converged, "grad {}", report.grad);
    assert!(report.energy < 1e-12, "energy {}", report.energy);
    assert!(x.iter().all(|v| v.abs() < 1e-5), "{x:?}");
}
