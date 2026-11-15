//! L-BFGS should take fewer Rosenbrock steps than steepest descent.

use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Control, LineSearch, Method, minimize_method};

#[test]
fn lbfgs_fewer_steps_than_steepest_on_rosenbrock() {
    let obj = Rosenbrock::<2>::new();
    let ctrl = Control::default()
        .with_maxiter(200)
        .with_gtol(1e-8)
        .with_istep(0.1)
        .with_maxmove(None);
    let ls = LineSearch::brent(40, 1e-12);
    let lbfgs = minimize_method(&obj, array![-1.2, 1.0], &ctrl, Method::lbfgs(), ls).unwrap();
    let sd = minimize_method(&obj, array![-1.2, 1.0], &ctrl, Method::Steepest, ls).unwrap();
    assert!(lbfgs.value < 1e-8);
    assert!(
        lbfgs.steps < sd.steps,
        "L-BFGS steps {} vs steepest {}",
        lbfgs.steps,
        sd.steps
    );
}
