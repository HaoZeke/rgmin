//! Persistent limited-memory BFGS (Nocedal-Wright 7.4, scaling 7.20).
//!
//! The production local method is L-BFGS with the strong Wolfe conditions
//! (Nocedal-Wright algorithms 3.5 and 3.6). Armijo alone accepts a step
//! that decreases the value without measuring curvature, and every later
//! direction is built from the stored pairs, so one bad pair degrades the
//! whole memory.
//!
//! Liu and Nocedal, *On the limited memory BFGS method for large scale
//! optimization*, <https://doi.org/10.1007/BF01589116>.
//! Nocedal, *Updating quasi-Newton matrices with limited storage*,
//! <https://doi.org/10.1090/s0025-5718-1980-0572855-7>.
//! Nocedal and Wright, *Numerical Optimization*,
//! <https://doi.org/10.1007/978-0-387-40065-5>.
//! Wolfe, *Convergence Conditions for Ascent Methods*,
//! <https://doi.org/10.1137/1011036>.

use std::cell::RefCell;

use ndarray::{Array1, ArrayView1};

use crate::control::{Control, StopNorm};
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::qn::solve_dense;
use crate::report::Report;
use crate::step::{finite, l2, next_istep, start_answer, take_step_reuse};
use eindir_core::{DifferentiableObjective, Objective};

/// How [`Lbfgs`] compares the gradient to [`Lbfgs::gtol`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradNorm {
    /// Euclidean `||g||_2`. Matches [`crate::minimize_lbfgs`].
    Euclidean,
    /// Infinity `||g||_∞`. Matches SciPy L-BFGS-B and hopping polish.
    Infinity,
}

/// `factr` that the L-BFGS-B 3.0 `setulb` comments give for moderate
/// accuracy (`lbfgsb.f`, Morales and Nocedal 2011,
/// <https://doi.org/10.1145/2049662.2049669>).
pub const LBFGSB_FACTR: f64 = 1.0e7;

/// The relative-decrease tolerance `factr * eps` of L-BFGS-B, SciPy's
/// default `ftol` for `method="L-BFGS-B"`, about 2.22e-9.
pub const LBFGSB_FTOL: f64 = LBFGSB_FACTR * f64::EPSILON;

/// The value and slope of a line-search probe, `(+inf, NaN)` when the
/// objective answered with a non-finite value or gradient. Every comparison
/// against NaN is false, so a raw NaN would fail no decrease test.
fn probed(f: f64, d: &Array1<f64>, g: &Array1<f64>) -> (f64, f64) {
    if finite(f, g.view()) {
        (f, d.dot(g))
    } else {
        (f64::INFINITY, f64::NAN)
    }
}

/// Stored curvature pair from one accepted step.
#[derive(Clone)]
struct Pair {
    s: Array1<f64>,
    y: Array1<f64>,
    rho: f64,
}

/// What replaces `q` between the two loops.
enum Middle<'a> {
    /// Dense `P^{-1} q`, or γ `q` when the solve fails.
    Precon(&'a ndarray::Array2<f64>),
    /// `exp.solve(q)`, or γ `q` when the solve fails.
    Exp(&'a crate::precon::Exp),
    /// γ `q` from the newest pair. No scaling when the memory is empty.
    Gamma,
    /// `exp.solve(q)` when it succeeds, otherwise `h0 * q`.
    Scaled {
        exp: Option<&'a crate::precon::Exp>,
        h0: f64,
    },
}

impl<'a> Middle<'a> {
    fn from_parts(
        precon: Option<&'a ndarray::Array2<f64>>,
        exp: Option<&'a crate::precon::Exp>,
    ) -> Self {
        if let Some(pmat) = precon {
            Self::Precon(pmat)
        } else if let Some(exp) = exp {
            Self::Exp(exp)
        } else {
            Self::Gamma
        }
    }
}

/// γ `q` with γ = `(s·y) / (y·y)` of the newest pair.
fn scale_gamma_into(memory: &[Pair], q: &mut Array1<f64>) {
    if let Some(p) = memory.last() {
        let yy = p.y.dot(&p.y);
        if yy > 0.0 {
            *q *= p.s.dot(&p.y) / yy;
        }
    }
}

/// Curvature saved by [`Lbfgs::curvature`].
pub(crate) struct Curvature {
    memory: Vec<Pair>,
    last: (f64, f64, bool),
    stored: bool,
}

/// L-BFGS whose curvature pairs survive between calls.
///
/// A hopping chain relaxes thousands of times from perturbations of an
/// already-relaxed structure. The curvature at a new start resembles the
/// curvature at the old minimum; a solver that forgets between calls pays
/// to rediscover it.
pub struct Lbfgs {
    memory: Vec<Pair>,
    /// Pairs retained; the usual choice is between five and ten.
    pub max_pairs: usize,
    /// Gradient-norm threshold that ends a relaxation.
    pub gtol: f64,
    /// Armijo sufficient-decrease constant, `c1` in the Wolfe conditions.
    pub armijo: f64,
    /// Curvature constant, `c2`. The usual choice for quasi-Newton is 0.9.
    pub curvature: f64,
    /// Line-search evaluations attempted before the direction is abandoned.
    pub max_line_evals: usize,
    /// Norm used against [`Lbfgs::gtol`].
    pub norm: GradNorm,
    /// Relative-decrease stop of the L-BFGS-B code (Zhu, Byrd, Lu and
    /// Nocedal, Algorithm 778, <https://doi.org/10.1145/279232.279236>).
    /// A line-search step from `f_k`
    /// to `f_{k+1}` ends [`Lbfgs::minimize`], [`Lbfgs::minimize_watched`]
    /// and [`Lbfgs::minimize_recognized`] at the new iterate when
    /// `f_k - f_{k+1} <= ftol * max(|f_k|, |f_{k+1}|, 1)`.
    /// Zero, the default, turns the test off, as does any value that is
    /// not positive. [`LBFGSB_FTOL`] is the SciPy default.
    pub ftol: f64,
    /// Al-Baali extra-updates: replay the newest pair this many extra times.
    pub extra_updates: usize,
    /// Li-Fukushima cautious `ε`. Zero disables the filter.
    pub cautious_eps: f64,
    /// Li-Fukushima cautious `α`.
    pub cautious_alpha: f64,
    /// When set, each direction is the HiGHS QP on the compact Hessian
    /// rather than the two-loop recursion.
    #[cfg(feature = "highs")]
    pub highs: Option<crate::lbfgs_qp::HighsStep>,
    /// \(H_0 = P^{-1}\) from an assembled exponential preconditioner.
    /// A dense matrix passed to [`Self::direction_with_precon`] is the
    /// pair / Lindh hook and wins over this factor.
    h0: Option<crate::precon::Exp>,
    /// `s·y` of the latest geometry step, stored even when the pair is dropped.
    pub(crate) last_sy: f64,
    /// `y·y` of the latest geometry step.
    pub(crate) last_yy: f64,
    /// [`Self::last_sy`] and [`Self::last_yy`] belong to a real step.
    pub(crate) have_curvature: bool,
    /// The latest [`Self::push_pair`] stored its pair.
    stored: bool,
    /// Reused two-loop, probe and line-search buffers. Boxed so the
    /// session enum does not carry them inline.
    work: Box<Work>,
}

/// Scratch that outlives one iteration. Each buffer is its own cell so
/// a direction and a pair can be borrowed together.
struct Work {
    /// Two-loop vector.
    q: RefCell<Array1<f64>>,
    /// `α` coefficients of the two-loop.
    alpha: RefCell<Vec<f64>>,
    /// Pair indices replayed by the two-loop, including extra updates.
    idxs: RefCell<Vec<usize>>,
    /// Line-search probe `x + a d` inside [`Lbfgs::line_search`].
    probe: RefCell<Array1<f64>>,
    /// `s` of the pair being stored.
    s_buf: RefCell<Array1<f64>>,
    /// `y` of the pair being stored.
    y_buf: RefCell<Array1<f64>>,
    /// Point and gradient the current outer step started from.
    old_x: RefCell<Array1<f64>>,
    old_g: RefCell<Array1<f64>>,
    /// Buffers for [`Lbfgs::step_objective`]'s line search.
    search: crate::workspace::SearchBuffers,
}

impl Work {
    fn new() -> Self {
        Self {
            q: RefCell::new(Array1::zeros(0)),
            alpha: RefCell::new(Vec::new()),
            idxs: RefCell::new(Vec::new()),
            probe: RefCell::new(Array1::zeros(0)),
            s_buf: RefCell::new(Array1::zeros(0)),
            y_buf: RefCell::new(Array1::zeros(0)),
            old_x: RefCell::new(Array1::zeros(0)),
            old_g: RefCell::new(Array1::zeros(0)),
            search: crate::workspace::SearchBuffers::default(),
        }
    }
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self::with_capacity(8)
    }
}

impl Lbfgs {
    /// Fresh solver with room for `max_pairs` curvature pairs.
    pub fn with_capacity(max_pairs: usize) -> Self {
        Self {
            memory: Vec::new(),
            max_pairs: max_pairs.max(1),
            gtol: 1e-6,
            armijo: 1e-4,
            curvature: 0.9,
            max_line_evals: 20,
            norm: GradNorm::Infinity,
            ftol: 0.0,
            extra_updates: 0,
            cautious_eps: 0.0,
            cautious_alpha: 0.01,
            #[cfg(feature = "highs")]
            highs: None,
            h0: None,
            last_sy: 0.0,
            last_yy: 0.0,
            have_curvature: false,
            stored: false,
            work: Box::new(Work::new()),
        }
    }

    /// Install or clear the exponential \(H_0 = P^{-1}\).
    pub(crate) fn set_h0(&mut self, h0: Option<crate::precon::Exp>) {
        self.h0 = h0;
    }

    /// Discards the stored curvature.
    ///
    /// Called when the chain moves somewhere structurally different, where
    /// the retained pairs describe a Hessian that no longer applies.
    pub fn forget(&mut self) {
        self.memory.clear();
        self.stored = false;
        self.last_sy = 0.0;
        self.last_yy = 0.0;
        self.have_curvature = false;
    }

    /// The pairs and latest geometry curvature, for [`Self::restore`].
    pub(crate) fn curvature(&self) -> Curvature {
        Curvature {
            memory: self.memory.clone(),
            last: (self.last_sy, self.last_yy, self.have_curvature),
            stored: self.stored,
        }
    }

    /// Puts back what [`Self::curvature`] saved.
    pub(crate) fn restore(&mut self, saved: Curvature) {
        self.memory = saved.memory;
        (self.last_sy, self.last_yy, self.have_curvature) = saved.last;
        self.stored = saved.stored;
    }

    /// Pairs currently held.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// True when no curvature is stored.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Drops oldest pairs when `max_pairs` shrank.
    pub fn trim(&mut self) {
        let cap = self.max_pairs.max(1);
        self.max_pairs = cap;
        while self.memory.len() > cap {
            self.memory.remove(0);
        }
    }

    pub(crate) fn search_direction(&self, x: ArrayView1<f64>, g: ArrayView1<f64>) -> Array1<f64> {
        self.fill_search_direction(x, g);
        self.work.q.borrow().clone()
    }

    /// Writes the search direction into [`Self::q`] without cloning it out.
    fn fill_search_direction(&self, x: ArrayView1<f64>, g: ArrayView1<f64>) {
        #[cfg(feature = "highs")]
        if self.highs.is_some()
            && let Ok(d) = self.highs_step(x, g)
        {
            let mut q = self.work.q.borrow_mut();
            crate::workspace::assign_fit(&mut q, d.view());
            return;
        }
        let _ = x;
        self.fill_direction(g, Middle::from_parts(None, self.h0.as_ref()));
    }

    /// Two-loop recursion: applies the inverse-Hessian approximation to `g`.
    ///
    /// An installed exponential preconditioner is \(H_0 = P^{-1}\).
    pub(crate) fn direction(&self, g: ArrayView1<f64>) -> Array1<f64> {
        self.direction_with_sources(g, None, self.h0.as_ref())
    }

    /// Two-loop with optional \(H_0 = P^{-1}\). `precon` is the pair / Lindh
    /// matrix \(P\); the middle product is `solve(P, q)`, not a scalar γ.
    /// A dense `precon` wins over an installed exponential matrix.
    pub(crate) fn direction_with_precon(
        &self,
        g: ArrayView1<f64>,
        precon: Option<&ndarray::Array2<f64>>,
    ) -> Array1<f64> {
        self.direction_with_sources(g, precon, None)
    }

    fn direction_with_sources(
        &self,
        g: ArrayView1<f64>,
        precon: Option<&ndarray::Array2<f64>>,
        exp: Option<&crate::precon::Exp>,
    ) -> Array1<f64> {
        self.fill_direction(g, Middle::from_parts(precon, exp));
        self.work.q.borrow().clone()
    }

    /// Two-loop into [`Self::q`]. The vector stays in the solver.
    fn fill_direction(&self, g: ArrayView1<f64>, middle: Middle<'_>) {
        {
            let mut q = self.work.q.borrow_mut();
            crate::workspace::assign_fit(&mut q, g);
        }
        self.apply_two_loop(middle);
    }

    fn apply_two_loop(&self, middle: Middle<'_>) {
        let m = self.memory.len();
        let mut idxs = self.work.idxs.borrow_mut();
        idxs.clear();
        idxs.extend(0..m);
        for _ in 0..self.extra_updates {
            if m > 0 {
                idxs.push(m - 1);
            }
        }
        let nidx = idxs.len();
        let mut alpha = self.work.alpha.borrow_mut();
        if alpha.len() < nidx {
            alpha.resize(nidx, 0.0);
        }
        let mut q = self.work.q.borrow_mut();
        for k in (0..nidx).rev() {
            let i = idxs[k];
            let p = &self.memory[i];
            let a = p.rho * crate::vecops::dot(p.s.view(), q.view());
            alpha[k] = a;
            crate::vecops::axpy(-a, p.y.view(), &mut q);
        }
        match middle {
            Middle::Precon(pmat) => match solve_dense(pmat, &q) {
                Some(solved) => q.assign(&solved),
                None => scale_gamma_into(&self.memory, &mut q),
            },
            Middle::Exp(exp) => match exp.solve(q.view()) {
                Some(solved) => q.assign(&solved),
                None => scale_gamma_into(&self.memory, &mut q),
            },
            Middle::Gamma => scale_gamma_into(&self.memory, &mut q),
            Middle::Scaled { exp, h0 } => {
                if let Some(solved) = exp.and_then(|exp| exp.solve(q.view())) {
                    q.assign(&solved);
                } else {
                    *q *= h0;
                }
            }
        }
        for k in 0..nidx {
            let i = idxs[k];
            let p = &self.memory[i];
            let b = p.rho * crate::vecops::dot(p.y.view(), q.view());
            crate::vecops::axpy(alpha[k] - b, p.s.view(), &mut q);
        }
        q.mapv_inplace(|v| -v);
    }

    /// `P^{-1} v` when an exponential preconditioner is installed and the solve succeeds.
    pub(crate) fn exp_apply(&self, v: ArrayView1<f64>) -> Option<Array1<f64>> {
        self.h0.as_ref().and_then(|exp| exp.solve(v))
    }

    /// Two-loop with a caller-supplied initial inverse curvature `h0`.
    /// Empty memory returns `-h0 g`. An installed exponential
    /// preconditioner replaces that scalar: the middle product is
    /// `P^{-1} q`, and a failed solve falls back to `h0`.
    pub(crate) fn direction_h0(&self, g: ArrayView1<f64>, h0: f64) -> Array1<f64> {
        self.fill_direction(
            g,
            Middle::Scaled {
                exp: self.h0.as_ref(),
                h0,
            },
        );
        self.work.q.borrow().clone()
    }

    /// Remember `s·y` and `y·y` from the step just taken.
    pub(crate) fn note_curvature(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        let sy = s.dot(y);
        let yy = y.dot(y);
        if sy.is_finite() && yy.is_finite() {
            self.last_sy = sy;
            self.last_yy = yy;
            self.have_curvature = true;
        } else {
            self.have_curvature = false;
        }
    }

    /// Store a geometry-step pair. Negative or vanishing curvature is dropped
    /// here; the caller resets memory from [`Self::note_curvature`].
    pub(crate) fn push_geometry(&mut self, s: Array1<f64>, y: Array1<f64>) {
        let sy = s.dot(&y);
        if !sy.is_finite() || sy.abs() < 1e-30 || sy <= 0.0 {
            return;
        }
        self.load_sy(s, y);
        self.install_pair(sy, false);
    }

    /// Records an accepted curvature pair (`s = x+ - x`, `y = g+ - g`).
    pub fn record(&mut self, s: Array1<f64>, y: Array1<f64>) {
        self.push(s, y);
    }

    /// Two-loop direction `d = −H g` (Nocedal-Wright 7.4).
    pub fn two_loop(&self, g: ArrayView1<f64>) -> Array1<f64> {
        self.direction(g)
    }

    /// Move every retained pair to the tangent space of the next iterate.
    pub(crate) fn transport<F>(&mut self, mut map: F)
    where
        F: FnMut(&Array1<f64>) -> Array1<f64>,
    {
        let pairs = std::mem::take(&mut self.memory);
        let stored = self.stored;
        let mut kept = false;
        for pair in pairs {
            kept = self.push_pair(map(&pair.s), map(&pair.y), None);
        }
        // A newest pair the map dropped is no longer there to replace.
        self.stored = stored && kept;
    }

    /// [`Self::transport`] with a map that can fail. Every pair is mapped
    /// before any is replaced, so a failure leaves the memory as it was.
    pub(crate) fn try_transport<F, E>(&mut self, mut map: F) -> std::result::Result<(), E>
    where
        F: FnMut(&Array1<f64>) -> std::result::Result<Array1<f64>, E>,
    {
        let moved = self
            .memory
            .iter()
            .map(|pair| Ok((map(&pair.s)?, map(&pair.y)?)))
            .collect::<std::result::Result<Vec<_>, E>>()?;
        let stored = self.stored;
        self.memory.clear();
        let mut kept = false;
        for (s, y) in moved {
            kept = self.push_pair(s, y, None);
        }
        self.stored = stored && kept;
        Ok(())
    }

    /// Stored pairs, oldest first.
    #[cfg(test)]
    pub(crate) fn pairs(&self) -> impl Iterator<Item = (&Array1<f64>, &Array1<f64>)> {
        self.memory.iter().map(|p| (&p.s, &p.y))
    }

    pub(crate) fn push(&mut self, s: Array1<f64>, y: Array1<f64>) -> bool {
        self.push_pair(s, y, None)
    }

    /// Replace the newest pair after a manifold retract. Only a pair the
    /// latest push stored is popped; a dropped pair leaves the previous
    /// one in place.
    pub(crate) fn replace_newest(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) {
        if self.stored {
            let _ = self.memory.pop();
        }
        self.push_pair(s, y, gnorm);
    }

    /// Store `(s, y)` unless the cautious or relative-curvature test
    /// drops it. Returns whether it was stored.
    pub(crate) fn push_pair(&mut self, s: Array1<f64>, y: Array1<f64>, gnorm: Option<f64>) -> bool {
        self.load_sy(s, y);
        self.store_loaded(gnorm)
    }

    /// Copy `s` and `y` into the pair scratch. A later store reuses the
    /// oldest pair's allocation once `max_pairs` is full.
    fn load_sy(&self, s: Array1<f64>, y: Array1<f64>) {
        {
            let mut buf = self.work.s_buf.borrow_mut();
            crate::workspace::assign_fit(&mut buf, s.view());
        }
        let mut buf = self.work.y_buf.borrow_mut();
        crate::workspace::assign_fit(&mut buf, y.view());
    }

    fn store_loaded(&mut self, gnorm: Option<f64>) -> bool {
        let sy = {
            let s = self.work.s_buf.borrow();
            let y = self.work.y_buf.borrow();
            let sy = s.dot(&*y);
            let sn = s.iter().map(|v| v * v).sum::<f64>().sqrt();
            let yn = y.iter().map(|v| v * v).sum::<f64>().sqrt();
            let ss = sn * sn;
            if self.cautious_eps > 0.0
                && let Some(g) = gnorm
            {
                let thresh = self.cautious_eps * ss * g.max(1.0e-30).powf(self.cautious_alpha);
                if sy < thresh {
                    drop(s);
                    drop(y);
                    self.stored = false;
                    return false;
                }
            }
            // Relative curvature: a tiny accepted trust step makes the
            // compact Hessian indefinite and HiGHS's QP solver does not return.
            if !sy.is_finite() || sy <= 1e-8 * sn * yn {
                drop(s);
                drop(y);
                self.stored = false;
                return false;
            }
            sy
        };
        self.install_pair(sy, true);
        true
    }

    /// Put the loaded `(s, y)` into memory with `rho = 1/sy`.
    ///
    /// At capacity the oldest pair's buffers are overwritten and rotated
    /// to the newest slot, which is what [`Self::trim`] did after a push.
    fn install_pair(&mut self, sy: f64, set_stored: bool) {
        let rho = 1.0 / sy;
        let cap = self.max_pairs.max(1);
        self.trim();
        if self.memory.len() >= cap {
            let mut pair = self.memory.remove(0);
            {
                let mut s = self.work.s_buf.borrow_mut();
                let mut y = self.work.y_buf.borrow_mut();
                std::mem::swap(&mut pair.s, &mut s);
                std::mem::swap(&mut pair.y, &mut y);
            }
            pair.rho = rho;
            self.memory.push(pair);
        } else {
            let s = {
                let mut s = self.work.s_buf.borrow_mut();
                let n = s.len();
                std::mem::replace(&mut *s, Array1::zeros(n))
            };
            let y = {
                let mut y = self.work.y_buf.borrow_mut();
                let n = y.len();
                std::mem::replace(&mut *y, Array1::zeros(n))
            };
            self.memory.push(Pair { s, y, rho });
            self.trim();
        }
        if set_stored {
            self.stored = true;
        }
    }

    fn gnorm(&self, g: &Array1<f64>) -> f64 {
        // A non-finite component means the gradient is broken, not small:
        // f64::max returns its other operand against NaN, so the infinity
        // fold would report an all-NaN gradient as norm zero and the
        // relaxation would terminate as converged at a garbage point. The
        // Euclidean arm failed safe only by accident, NaN propagating into
        // a comparison that then never passes. Both arms now answer
        // infinity, which no gtol accepts; the seam's infinity norm
        // carries that guarantee itself.
        match self.norm {
            GradNorm::Euclidean => {
                if g.iter().any(|v| !v.is_finite()) {
                    return f64::INFINITY;
                }
                l2(g)
            }
            GradNorm::Infinity => crate::vecops::nrminf(g.view()),
        }
    }

    /// The step from `f_old` to `f_new` passes the [`Lbfgs::ftol`] test
    /// (`lbfgsb.f`, `mainlb`).
    fn stalled(&self, f_old: f64, f_new: f64) -> bool {
        self.ftol > 0.0 && f_old - f_new <= self.ftol * f_old.abs().max(f_new.abs()).max(1.0)
    }

    /// Strong Wolfe line search by bracketing then cubic-interpolated zoom.
    ///
    /// Nocedal and Wright algorithms 3.5 and 3.6. Returns whether a step
    /// was accepted and how many evaluations it cost. A probe answered with
    /// a non-finite value or gradient fails the decrease test, so it caps
    /// the bracket and the zoom bisects back toward the accepted end.
    fn line_search<F>(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        slope: f64,
        fg: &mut F,
    ) -> (bool, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let f0 = *f;
        let mut evals = 0usize;

        let mut a_prev = 0.0;
        let mut f_prev = f0;
        let mut slope_prev = slope;
        // A quasi-Newton direction already carries the step length. Nocedal
        // and Wright require that alpha = 1 is tried first. With no memory
        // the direction is the raw negative gradient and needs a length.
        let mut a = if self.memory.is_empty() {
            let q = self.work.q.borrow();
            let dnorm = q.iter().fold(0.0_f64, |acc, v| acc + v * v).sqrt();
            if dnorm > 1.0 { 1.0 / dnorm } else { 1.0 }
        } else {
            1.0
        };
        let mut lo = 0.0;
        let mut f_lo = f0;
        let mut slope_lo = slope;
        let mut hi = f64::NAN;
        let mut f_hi = f64::NAN;
        let mut slope_hi = f64::NAN;
        let mut bracketed = false;

        let bracket_cap = self.max_line_evals;
        let total_cap = 2 * self.max_line_evals;

        for i in 0..bracket_cap {
            let (fa, ga) = match self.probe_at(a, x, fg, &mut evals) {
                Some(v) => v,
                None => return (false, evals),
            };
            let (fa, slope_a) = {
                let q = self.work.q.borrow();
                probed(fa, &q, &ga)
            };
            if fa > f0 + self.armijo * a * slope || (i > 0 && fa >= f_prev) {
                lo = a_prev;
                f_lo = f_prev;
                slope_lo = slope_prev;
                hi = a;
                f_hi = fa;
                slope_hi = slope_a;
                bracketed = true;
                break;
            }
            if slope_a.abs() <= -self.curvature * slope {
                self.accept(x, f, g, a, (fa, ga));
                return (true, evals);
            }
            if slope_a >= 0.0 {
                lo = a;
                f_lo = fa;
                slope_lo = slope_a;
                hi = a_prev;
                f_hi = f_prev;
                slope_hi = slope_prev;
                bracketed = true;
                break;
            }
            a_prev = a;
            f_prev = fa;
            slope_prev = slope_a;
            a *= 2.0;
        }

        if !bracketed {
            return (false, evals);
        }

        while evals < total_cap {
            let width = hi - lo;
            let mut trial = lo + 0.5 * width;
            // Cubic Hermite minimizer over the bracket (Nocedal-Wright
            // eq. 3.59), possible because both ends carry their slopes:
            // the slope at hi is information an evaluation already paid
            // for, and discarding it forced a quadratic model that the
            // doc nevertheless called cubic. More-Thuente's dcstep is
            // the reference for the guards: the discriminant clamps at
            // zero, a degenerate denominator falls back, and every
            // candidate is confined to the bracket interior so a wild
            // extrapolation costs a bisection, never a divergence.
            if slope_hi.is_finite() {
                let d1 = slope_lo + slope_hi - 3.0 * (f_lo - f_hi) / (lo - hi);
                let disc = d1 * d1 - slope_lo * slope_hi;
                if disc >= 0.0 && (lo - hi).abs() > 1e-16 {
                    let d2 = (hi - lo).signum() * disc.sqrt();
                    let denom = slope_hi - slope_lo + 2.0 * d2;
                    if denom.abs() > 1e-16 {
                        let q = hi - (hi - lo) * (slope_hi + d2 - d1) / denom;
                        if (q - lo) / width > 0.1 && (q - lo) / width < 0.9 {
                            trial = q;
                        }
                    }
                }
            } else {
                let denom = 2.0 * (f_hi - f_lo - slope_lo * width);
                if denom.abs() > 1e-16 {
                    let q = lo - slope_lo * width * width / denom;
                    if (q - lo) / width > 0.1 && (q - lo) / width < 0.9 {
                        trial = q;
                    }
                }
            }
            let (ft, gt) = match self.probe_at(trial, x, fg, &mut evals) {
                Some(v) => v,
                None => return (false, evals),
            };
            let (ft, slope_t) = {
                let q = self.work.q.borrow();
                probed(ft, &q, &gt)
            };
            if ft > f0 + self.armijo * trial * slope || ft >= f_lo {
                hi = trial;
                f_hi = ft;
                slope_hi = slope_t;
            } else {
                if slope_t.abs() <= -self.curvature * slope {
                    self.accept(x, f, g, trial, (ft, gt));
                    return (true, evals);
                }
                if slope_t * (hi - lo) >= 0.0 {
                    hi = lo;
                    f_hi = f_lo;
                    slope_hi = slope_lo;
                }
                lo = trial;
                f_lo = ft;
                slope_lo = slope_t;
            }
            if (hi - lo).abs() < 1e-14 {
                break;
            }
        }
        (false, evals)
    }

    fn probe_at<F>(
        &self,
        a: f64,
        x: &Array1<f64>,
        fg: &mut F,
        evals: &mut usize,
    ) -> Option<(f64, Array1<f64>)>
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        let mut scratch = self.work.probe.borrow_mut();
        crate::workspace::assign_fit(&mut scratch, x.view());
        {
            let q = self.work.q.borrow();
            scratch.scaled_add(a, &*q);
        }
        let r = fg(scratch.view());
        if r.is_some() {
            *evals += 1;
        }
        r
    }

    fn accept(
        &mut self,
        x: &mut Array1<f64>,
        f: &mut f64,
        g: &mut Array1<f64>,
        step: f64,
        new: (f64, Array1<f64>),
    ) {
        let (f_new, g_new) = new;
        {
            let q = self.work.q.borrow();
            let mut s = self.work.s_buf.borrow_mut();
            crate::workspace::assign_fit(&mut s, q.view());
            *s *= step;
        }
        {
            let mut y = self.work.y_buf.borrow_mut();
            crate::workspace::assign_fit(&mut y, g_new.view());
            *y -= &*g;
        }
        let _ = self.store_loaded(None);
        {
            let q = self.work.q.borrow();
            x.scaled_add(step, &*q);
        }
        *f = f_new;
        *g = g_new;
    }

    /// Relaxes `x0`, calling `fg` for value and gradient.
    ///
    /// `fg` returns `None` when the caller's budget is spent, which ends the
    /// relaxation where it stands. Returns the value, the point, and the
    /// number of evaluations used. A non-finite value or gradient at a
    /// trial step shortens the step; at `x0` it ends the relaxation there
    /// with value `+inf`, as a spent budget does.
    pub fn minimize<F>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        fg: F,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.minimize_watched(x0, max_iter, fg, |_, _| true)
    }

    /// Relaxes `x0`, offering each accepted iterate to `watch`.
    ///
    /// `watch` receives the iteration index and the value at that iterate,
    /// and returning `false` ends the relaxation there.
    pub fn minimize_watched<F, W>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
        mut watch: W,
    ) -> (f64, Array1<f64>, usize)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        W: FnMut(usize, f64) -> bool,
    {
        self.trim();
        let mut x = x0.to_owned();
        let mut evals = 0usize;
        let (mut f, mut g) = match fg(x.view()) {
            Some(v) => v,
            None => return (f64::INFINITY, x, evals),
        };
        evals += 1;
        if !finite(f, g.view()) {
            return (f64::INFINITY, x, evals);
        }

        let mut stalled = false;
        for it in 0..max_iter {
            if !watch(it, f) {
                break;
            }
            if stalled || self.gnorm(&g) < self.gtol {
                break;
            }
            let f_old = f;
            self.fill_search_direction(x.view(), g.view());
            let slope = self.work.q.borrow().dot(&g);
            if !slope.is_finite() && self.memory.is_empty() {
                break;
            }
            if slope >= 0.0 || !slope.is_finite() {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, slope, &mut fg);
            evals += evals_used;
            stalled = ok && self.stalled(f_old, f);
            if !ok {
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
            }
        }
        (f, x, evals)
    }

    /// Relaxes `x0`, consulting `recognise` at each accepted iterate.
    ///
    /// `recognise` maps an accepted iterate to a stand-in result when the
    /// caller can certify where this descent ends: a minimum already on
    /// file whose catchment the iterate has entered. Returning
    /// `Some((f_known, x_known))` ends the relaxation with that result and
    /// the evaluations spent so far, which is the refund -- the remainder
    /// of a descent whose outcome is already known is never paid for. The
    /// final `bool` reports whether the result is a stand-in, so a caller
    /// auditing its recogniser can tell refunded descents from completed
    /// ones.
    ///
    /// The hook sits where `watch` sits, at accepted iterates only, so a
    /// recogniser never sees a trial step the optimizer has not adopted.
    /// Soundness is the caller's contract: the stand-in must be the
    /// minimum this descent would have reached, and the tolerance for
    /// getting that wrong belongs to the caller's error budget, not to
    /// the solver.
    pub fn minimize_recognized<F, R>(
        &mut self,
        x0: ArrayView1<f64>,
        max_iter: usize,
        mut fg: F,
        mut recognise: R,
    ) -> (f64, Array1<f64>, usize, bool)
    where
        F: FnMut(ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
        R: FnMut(usize, f64, ArrayView1<f64>) -> Option<(f64, Array1<f64>)>,
    {
        self.trim();
        let mut x = x0.to_owned();
        let mut evals = 0usize;
        let (mut f, mut g) = match fg(x.view()) {
            Some(v) => v,
            None => return (f64::INFINITY, x, evals, false),
        };
        evals += 1;
        if !finite(f, g.view()) {
            return (f64::INFINITY, x, evals, false);
        }

        let mut stalled = false;
        for it in 0..max_iter {
            if let Some((f_known, x_known)) = recognise(it, f, x.view()) {
                return (f_known, x_known, evals, true);
            }
            if stalled || self.gnorm(&g) < self.gtol {
                break;
            }
            let f_old = f;
            self.fill_search_direction(x.view(), g.view());
            let slope = self.work.q.borrow().dot(&g);
            if !slope.is_finite() && self.memory.is_empty() {
                break;
            }
            if slope >= 0.0 || !slope.is_finite() {
                self.forget();
                continue;
            }
            let (ok, evals_used) = self.line_search(&mut x, &mut f, &mut g, slope, &mut fg);
            evals += evals_used;
            stalled = ok && self.stalled(f_old, f);
            if !ok {
                if self.memory.is_empty() {
                    break;
                }
                self.forget();
            }
        }
        (f, x, evals, false)
    }

    /// Cold-start L-BFGS over an eindir objective, any [`LineSearch`].
    ///
    /// Uses Euclidean `||g||_2` against `control.gtol` so
    /// [`crate::minimize_lbfgs`] keeps its existing reports.
    pub fn minimize_objective<O>(
        &mut self,
        obj: &O,
        init: impl Into<Array1<f64>>,
        control: &Control,
        linesearch: LineSearch,
    ) -> Result<Report>
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let mut pos = init.into();
        if pos.len() != Objective::dim(obj) {
            return Err(Error::Dim {
                got: pos.len(),
                dim: Objective::dim(obj),
            });
        }
        pos = obj.bounds().clip(pos.view());
        let (mut value, mut grad) = start_answer(obj, &pos)?;
        let mut istep = control.istep;

        for step in 0..control.maxiter {
            let gnorm = l2(&grad);
            if gnorm < control.gtol {
                return Ok(Report {
                    value,
                    coords: pos,
                    steps: step,
                    grad_norm: gnorm,
                    stop_norm: StopNorm::Two,
                    grad: None,
                });
            }
            self.step_objective(
                obj, &mut pos, &mut value, &mut grad, &mut istep, linesearch, control,
            );
        }
        Ok(Report {
            value,
            coords: pos,
            steps: control.maxiter,
            grad_norm: l2(&grad),
            stop_norm: StopNorm::Two,
            grad: None,
        })
    }

    /// One outer L-BFGS iteration: two-loop direction, line search, pair.
    #[expect(
        clippy::too_many_arguments,
        reason = "public signature; grouping the arguments would break callers"
    )]
    pub fn step_objective<O>(
        &mut self,
        obj: &O,
        pos: &mut Array1<f64>,
        value: &mut f64,
        grad: &mut Array1<f64>,
        istep: &mut f64,
        linesearch: LineSearch,
        control: &Control,
    ) where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        self.fill_search_direction(pos.view(), grad.view());
        {
            let mut old = self.work.old_x.borrow_mut();
            crate::workspace::assign_fit(&mut old, pos.view());
        }
        {
            let mut old = self.work.old_g.borrow_mut();
            crate::workspace::assign_fit(&mut old, grad.view());
        }
        // Curvature history supplies the quasi-Newton step length. Wolfe
        // starts at the unit step so accepted steps retain superlinear
        // convergence instead of inheriting repeated line-search shrinkage.
        let trial_step = if self.memory.is_empty() { *istep } else { 1.0 };
        let (mut lsstep, mut moved) =
            self.search_from_q(obj, pos, *value, trial_step, linesearch, control);
        if !moved && !self.memory.is_empty() {
            // Retained curvature can describe a different objective scale.
            // Retry with an empty history before reporting a stalled step.
            self.forget();
            self.fill_search_direction(pos.view(), grad.view());
            (lsstep, moved) =
                self.search_from_q(obj, pos, *value, control.istep, linesearch, control);
        }
        if moved {
            pos.assign(&self.work.search.clipped);
        }
        let ev = obj.value_and_gradient(pos.view());
        *value = ev.0;
        *grad = ev.1;
        if moved {
            self.stage_secant(pos, grad);
            let _ = self.store_loaded(None);
        }
        *istep = next_istep(lsstep, control);
    }

    /// Line search along [`Self::q`], then the clip and cap of `take_step`.
    fn search_from_q<O>(
        &mut self,
        obj: &O,
        pos: &Array1<f64>,
        value: f64,
        istep: f64,
        linesearch: LineSearch,
        control: &Control,
    ) -> (f64, bool)
    where
        O: DifferentiableObjective<f64> + ?Sized,
    {
        let dir = std::mem::take(&mut *self.work.q.borrow_mut());
        let outcome = take_step_reuse(
            obj,
            pos,
            value,
            dir.view(),
            istep,
            linesearch,
            control,
            &mut self.work.search,
        );
        *self.work.q.borrow_mut() = dir;
        outcome
    }

    /// `s = x - x_old`, `y = g - g_old` in the pair scratch.
    fn stage_secant(&self, pos: &Array1<f64>, grad: &Array1<f64>) {
        let old_x = self.work.old_x.borrow();
        let old_g = self.work.old_g.borrow();
        {
            let mut s = self.work.s_buf.borrow_mut();
            if s.len() != pos.len() {
                *s = pos - &*old_x;
            } else {
                s.assign(pos);
                *s -= &*old_x;
            }
        }
        let mut y = self.work.y_buf.borrow_mut();
        if y.len() != grad.len() {
            *y = grad - &*old_g;
        } else {
            y.assign(grad);
            *y -= &*old_g;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, array};

    use super::{LBFGSB_FTOL, Lbfgs};

    /// Above `|f| = 1` the threshold scales with `|f|`; below, it is
    /// `ftol` itself. A value that is not positive passes no step.
    #[test]
    fn the_decrease_test_is_relative_above_one_and_absolute_below() {
        let mut lbfgs = Lbfgs {
            ftol: LBFGSB_FTOL,
            ..Lbfgs::default()
        };
        let big = 1.0 / LBFGSB_FTOL;
        assert!(lbfgs.stalled(big, big - 0.5));
        assert!(!lbfgs.stalled(big, big - 2.0));
        let small = LBFGSB_FTOL;
        assert!(lbfgs.stalled(small, 0.5 * small));
        assert!(!lbfgs.stalled(small, -small));
        for ftol in [0.0, -LBFGSB_FTOL, f64::NAN] {
            lbfgs.ftol = ftol;
            assert!(!lbfgs.stalled(1.0, 1.0), "ftol {ftol}");
        }
    }

    fn held(lbfgs: &Lbfgs) -> Vec<(Array1<f64>, Array1<f64>)> {
        lbfgs.pairs().map(|(s, y)| (s.clone(), y.clone())).collect()
    }

    /// A transport that drops the newest pair leaves no pair of the latest
    /// push, so the next replacement keeps the older pair.
    #[test]
    fn a_transport_that_drops_the_newest_pair_keeps_the_older_one() {
        let mut lbfgs = Lbfgs::with_capacity(4);
        assert!(lbfgs.push(array![1.0, 0.0], array![2.0, 0.0]));
        assert!(lbfgs.push(array![1.0, 1.0], array![-1.0, 3.0]));
        // On the first coordinate alone the newer pair has s.y = -1.
        lbfgs.transport(|v| array![v[0], 0.0]);
        let older = (array![1.0, 0.0], array![2.0, 0.0]);
        assert_eq!(held(&lbfgs), vec![older.clone()]);
        lbfgs.replace_newest(array![0.0, 1.0], array![0.0, 3.0], None);
        assert_eq!(
            held(&lbfgs),
            vec![older, (array![0.0, 1.0], array![0.0, 3.0])]
        );
    }

    /// A map that fails on the second pair, after the first has moved,
    /// leaves both pairs as they were.
    #[test]
    fn a_failed_transport_keeps_every_pair() {
        let mut lbfgs = Lbfgs::with_capacity(4);
        assert!(lbfgs.push(array![1.0, 0.0], array![2.0, 0.0]));
        assert!(lbfgs.push(array![0.0, 1.0], array![0.0, 3.0]));
        let before = held(&lbfgs);
        let mut calls = 0;
        let moved = lbfgs.try_transport(|v| {
            calls += 1;
            if calls < 3 { Ok(v * 0.5) } else { Err(calls) }
        });
        assert_eq!(moved, Err(3));
        assert_eq!(held(&lbfgs), before);
        assert!(lbfgs.stored);
    }
    /// A fallible transport that succeeds but drops the newest pair also
    /// leaves the older pair to the next replacement.
    #[test]
    fn a_fallible_transport_that_drops_the_newest_pair_keeps_the_older_one() {
        let mut lbfgs = Lbfgs::with_capacity(4);
        assert!(lbfgs.push(array![1.0, 0.0], array![2.0, 0.0]));
        assert!(lbfgs.push(array![1.0, 1.0], array![-1.0, 3.0]));
        let moved: Result<(), ()> = lbfgs.try_transport(|v| Ok(array![v[0], 0.0]));
        assert_eq!(moved, Ok(()));
        let older = (array![1.0, 0.0], array![2.0, 0.0]);
        assert_eq!(held(&lbfgs), vec![older.clone()]);
        lbfgs.replace_newest(array![0.0, 1.0], array![0.0, 3.0], None);
        assert_eq!(
            held(&lbfgs),
            vec![older, (array![0.0, 1.0], array![0.0, 3.0])]
        );
    }
}
