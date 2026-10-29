//! Result of a minimization.

use ndarray::Array1;

/// Final point, value, and iteration count.
#[derive(Clone, Debug)]
pub struct Report {
    /// `f(x)` at [`Report::coords`].
    pub value: f64,
    /// Accepted coordinates.
    pub coords: Array1<f64>,
    /// Outer iterations performed.
    pub steps: usize,
    /// Gradient norm at the accepted point. A Euclidean session with a box
    /// and no additional constraints reports `||x - clip(x-∇f, lo, hi)||_2`.
    /// Other sessions report the norm of their projected or raw gradient.
    /// A session measures it in [`crate::StopNorm`], `||.||_2` by default.
    pub grad_norm: f64,
    /// Gradient that [`Report::grad_norm`] measures, before any box
    /// projection. Sessions fill it. One-shot drivers leave it `None`.
    pub grad: Option<Array1<f64>>,
}
