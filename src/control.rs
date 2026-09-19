//! Iteration limits and gradient tolerance.

/// Outer-loop controls for [`crate::minimize`].
#[derive(Clone, Debug)]
pub struct Control {
    /// Maximum CG iterations.
    pub maxiter: usize,
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// The opening step of a line search. Quasi-Newton methods (BFGS,
    /// L-BFGS, SR1, SR2) open every line search here, since their direction
    /// carries the scale and 1.0 is the natural trial; steepest descent,
    /// NLCG and Adam open here once and then at half the step last
    /// accepted.
    pub istep: f64,
    /// Optional Euclidean cap on a proposed step (xtsci `maxmove`).
    pub maxmove: Option<f64>,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            maxiter: 100,
            gtol: 1e-5,
            istep: 1.0,
            maxmove: None,
        }
    }
}
