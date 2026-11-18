//! Drive a `Solver` one step at a time, and stop when the host says so.
//!
//! `cargo run --example tutorial_session`

use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use rgmin::{Accept, Control, Method, Solver};

fn main() {
    let obj = Rosenbrock::<2>::new();
    let mut x = array![-1.2, 1.0];
    let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2);
    solver.set_accept(Accept::Energy);
    let mut steps = 0;
    loop {
        let rep = solver.step(&obj, &mut x).unwrap();
        steps += 1;
        if rep.grad_norm < 1e-4 || steps == 200 {
            break;
        }
    }
    assert!((x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3);
}
