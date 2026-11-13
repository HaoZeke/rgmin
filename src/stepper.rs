//! One outer iteration of a local minimizer.
//!
//! rgsaddle's band and minimum-mode sessions call [`Stepper::step`]
//! and own the loop. [`crate::Solver`] is the stepper. The production
//! unconstrained method is L-BFGS with the eOn cap
//! ([`crate::LbfgsStep::Capped`] at [`crate::EON_MAX_MOVE`]): no line
//! search. [`crate::Solver::set_lbfgs_step`]`(`[crate::LbfgsStep::Wolfe]`)`
//! opts into the strong Wolfe search.
//!
//! FIRE and FIRE 2.0 are [`crate::Method::Fire`]. Nonlinear CG is
//! [`crate::Method::Nlcg`]; the eOn coefficient and restart are
//! [`crate::Method::eon_cg`], and the line-search-free client step is
//! [`crate::minimize_eon_cg`]. The exponential preconditioner is
//! [`crate::Solver::set_exp`]: L-BFGS uses it as `H0`, and FIRE and CG
//! use it as the metric.
//!
//! ```
//! use ndarray::array;
//! use rgmin::{Control, Method, Oracle, Solver, Stepper};
//!
//! let obj = Oracle::unbounded(2, |x: ndarray::ArrayView1<f64>| {
//!     (x.dot(&x), ndarray::Array1::from_iter(x.iter().map(|v| 2.0 * v)))
//! });
//! let mut solver = Solver::new(Method::lbfgs(), Control::default(), 2);
//! let mut x = array![1.0, -0.5];
//! let report = Stepper::step(&mut solver, &obj, &mut x).unwrap();
//! assert!(report.grad_norm.is_finite());
//! assert!(x.iter().all(|v| v.abs() < 1.0));
//! ```

use eindir_core::DifferentiableObjective;
use ndarray::Array1;

use crate::Result;
use crate::report::Report;
use crate::session::Solver;

/// One outer iteration. The host owns `x` and decides when to stop.
pub trait Stepper {
    /// Advance `x` by one iteration of this method.
    ///
    /// A non-finite answer is [`crate::Error::Oracle`] and leaves `x`
    /// where it was. A dimension mismatch is [`crate::Error::Dim`].
    fn step<O>(&mut self, objective: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized;
}

impl Stepper for Solver {
    fn step<O>(&mut self, objective: &O, x: &mut Array1<f64>) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        Solver::step(self, objective, x)
    }
}
