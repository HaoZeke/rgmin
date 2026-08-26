//! Errors from a minimization.

use thiserror::Error;

/// Recoverable minimization failure.
#[derive(Debug, Error)]
pub enum Error {
    /// Initial point length does not match the objective dimension.
    #[error("init length {got} != objective dim {dim}")]
    Dim {
        /// Length of the supplied start vector.
        got: usize,
        /// `Objective::dim`.
        dim: usize,
    },
    /// HiGHS rejected the L-BFGS quadratic model.
    #[error("HiGHS: {0}")]
    Highs(String),
    /// Newton / RFO / dogleg needs a Hessian oracle.
    #[error("Newton/RFO/dogleg needs a Hessian; call step_hess")]
    NeedHessian,
    /// Packed manifold rejected this ambient dimension.
    #[error("{kind} rejected dimension {got}")]
    ManifoldDim {
        /// Token (`so3`, `se3`, `rigid_quotient`, `mw_rigid`).
        kind: &'static str,
        /// Length of the working vector.
        got: usize,
    },
    /// SCG cannot make progress (non-finite objective everywhere it
    /// can step, or damping at its limit).
    #[error("SCG stalled: {what}")]
    ScgStalled {
        /// What exhausted the algorithm.
        what: &'static str,
    },
    /// The trust region collapsed without an acceptable step: a
    /// non-finite gradient, a broken curvature action, or an
    /// objective that rejects every trial the model proposes.
    #[error("trust region collapsed after {steps} steps")]
    TrustCollapsed {
        /// Outer iterations completed when the radius hit its floor.
        steps: usize,
    },
    /// Host oracle returned a non-finite value or gradient.
    #[error("oracle: {what}")]
    Oracle {
        /// What the host callback failed to produce.
        what: &'static str,
    },
    /// Named eigensolver is not linked in this build. Fail closed:
    /// the waist does not silently fall back to Lanczos.
    #[error("eigensolver {kind} is not linked in this build")]
    EigenUnavailable {
        /// Closed-enum name (`elpa`, `primme`, ...).
        kind: &'static str,
    },
    /// Linked SLEPc EPS rejected the typed configuration or the pair.
    #[error("SLEPc: {what}")]
    Slepc {
        /// What the typed EPS/ST call failed to produce.
        what: &'static str,
    },
    /// Linked PRIMME `dprimme` rejected the typed configuration or the pair.
    #[error("PRIMME: {what}")]
    Primme {
        /// What the typed `primme_params` / `dprimme` call failed to produce.
        what: &'static str,
    },
    /// Named backend only computes the full spectrum. Partial `nev`
    /// is refused rather than silently solved as `n` and trimmed.
    #[error("eigensolver {kind} is full-spectrum only; nev {nev} < n {n}")]
    EigenFullSpectrum {
        /// Closed-enum name (`eigenExa`, ...).
        kind: &'static str,
        /// Requested pair count.
        nev: usize,
        /// Matrix order.
        n: usize,
    },
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
