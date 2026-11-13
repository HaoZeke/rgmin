//! A line search that never calls an objective.
//!
//! The caller already has the energy and the gradient. Each
//! [`SuppliedSearch::offer`] consumes that pair and either asks for the
//! next trial or returns the accepted step. The trials are the ones
//! [`super::LineSearch::search`] would make on the same answers.

use ndarray::{Array1, ArrayView1};

use super::conditions::{armijo, goldstein, strong_curvature};
use super::{LineSearch, axpy, scored};
use crate::step::finite;

/// What the caller does next.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchStep {
    /// Evaluate the objective at `x` (`x0 + alpha * direction`) and pass
    /// that energy and gradient to [`SuppliedSearch::offer`].
    Evaluate {
        /// Trial step length along the direction.
        alpha: f64,
        /// Trial point.
        x: Array1<f64>,
    },
    /// The search is finished. `x` is the accepted point, `value` its
    /// energy, and `alpha` the absolute step length
    /// [`LineSearch::search`] reports.
    Accepted {
        /// Accepted point.
        x: Array1<f64>,
        /// Energy at `x`.
        value: f64,
        /// Absolute step length.
        alpha: f64,
    },
}

/// Line search driven by a caller-supplied energy and gradient.
///
/// Nothing in this type calls an objective. The first [`Self::offer`] is
/// the energy and gradient at the start `x`. Later offers are the energy
/// and gradient at the trial from the previous [`SearchStep::Evaluate`].
///
/// ```
/// use ndarray::{Array1, array};
/// use rgmin::{LineSearch, SearchStep, SuppliedSearch};
///
/// fn quad(x: f64) -> (f64, f64) {
///     let z = x - 1.0;
///     (z * z, 2.0 * z)
/// }
/// let start = array![0.0];
/// let dir = array![1.0];
/// let mut search = SuppliedSearch::new(
///     LineSearch::Backtracking { c: 1e-4, beta: 0.5, maxiter: 8 },
///     start.view(),
///     dir.view(),
///     1.0,
/// );
/// let (f0, g0) = quad(0.0);
/// let mut step = search.offer(f0, array![g0].view());
/// let accepted = loop {
///     match step {
///         SearchStep::Evaluate { x, .. } => {
///             let (f, g) = quad(x[0]);
///             step = search.offer(f, array![g].view());
///         }
///         SearchStep::Accepted { x, .. } => break x,
///     }
/// };
/// assert!((accepted[0] - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct SuppliedSearch {
    x0: Array1<f64>,
    dir: Array1<f64>,
    istep: f64,
    kind: LineSearch,
    state: State,
}

#[derive(Clone, Debug)]
enum State {
    Fresh,
    Back(Back),
    Wolfe(Wolfe),
    Bracket(Bracket),
    Brent(Brent),
    Done {
        x: Array1<f64>,
        value: f64,
        alpha: f64,
    },
}

#[derive(Clone, Debug)]
struct Back {
    f0: f64,
    slope: f64,
    alpha: f64,
    best_x: Array1<f64>,
    best_f: f64,
    best_alpha: f64,
    c: f64,
    beta: f64,
    maxiter: usize,
    goldstein: bool,
    iter: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum WolfeAt {
    Outer,
    ZoomLo,
    ZoomTrial,
    FinishValue,
    FinishSlope,
}

#[derive(Clone, Debug)]
struct Wolfe {
    f0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
    alpha_max: f64,
    alpha_prev: f64,
    phi_prev: f64,
    alpha: f64,
    i: usize,
    best_x: Array1<f64>,
    best_f: f64,
    best_a: f64,
    at: WolfeAt,
    lo: f64,
    hi: f64,
    phi_lo: f64,
    zoom_iter: usize,
    t: f64,
    ft: f64,
}

#[derive(Clone, Debug)]
struct Bracket {
    f0: f64,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    c: f64,
    fc: f64,
    u: f64,
    it: usize,
    maxiter: usize,
    tol: f64,
    /// 0 = φ(a), 1 = φ(b), 2 = φ(c), 3 = φ(u).
    stage: u8,
}

#[derive(Clone, Debug)]
struct Brent {
    f0: f64,
    a: f64,
    b: f64,
    x: f64,
    w: f64,
    v: f64,
    fx: f64,
    fw: f64,
    fv: f64,
    e: f64,
    d: f64,
    u: f64,
    iter: usize,
    maxiter: usize,
    tol: f64,
    /// True until the midpoint value has been stored.
    init: bool,
}

const GOLD: f64 = 1.618_034;
const CGOLD: f64 = 0.381_966;

impl SuppliedSearch {
    /// `x` is the start and `dir` the search direction.
    ///
    /// A direction whose length is not `x`'s, or a non-finite direction,
    /// ends on the first offer at the start with step length zero.
    pub fn new(
        kind: LineSearch,
        x: ArrayView1<'_, f64>,
        dir: ArrayView1<'_, f64>,
        istep: f64,
    ) -> Self {
        Self {
            x0: x.to_owned(),
            dir: dir.to_owned(),
            istep,
            kind,
            state: State::Fresh,
        }
    }

    /// Consume the energy `f` and gradient `g` at the point the search is
    /// waiting on. The first call is the start. Later calls are the trial
    /// from [`SearchStep::Evaluate`].
    pub fn offer(&mut self, f: f64, g: ArrayView1<'_, f64>) -> SearchStep {
        if matches!(self.state, State::Done { .. }) {
            return self.accepted();
        }
        if matches!(self.state, State::Fresh) {
            return self.begin(f, g);
        }
        let f = self.scored(f, g);
        let slope = self.directional(g);
        match std::mem::replace(&mut self.state, State::Fresh) {
            State::Back(st) => self.back_offer(st, f),
            State::Wolfe(st) => self.wolfe_offer(st, f, slope),
            State::Bracket(st) => self.bracket_offer(st, f),
            State::Brent(st) => self.brent_offer(st, f),
            State::Done { x, value, alpha } => {
                self.state = State::Done { x, value, alpha };
                self.accepted()
            }
            State::Fresh => self.accepted(),
        }
    }

    fn begin(&mut self, f: f64, g: ArrayView1<'_, f64>) -> SearchStep {
        let bad = self.dir.len() != self.x0.len()
            || g.len() != self.x0.len()
            || !finite(f, g)
            || self.dir.iter().any(|v| !v.is_finite());
        if bad {
            return self.finish(self.x0.clone(), f, 0.0);
        }
        let slope = self.directional(g);
        match self.kind {
            LineSearch::Backtracking { c, beta, maxiter } => {
                if maxiter == 0 {
                    return self.finish(self.x0.clone(), f, 0.0);
                }
                let alpha = self.istep.max(1e-16);
                let st = Back {
                    f0: f,
                    slope,
                    alpha,
                    best_x: self.x0.clone(),
                    best_f: f,
                    best_alpha: 0.0,
                    c,
                    beta,
                    maxiter,
                    goldstein: false,
                    iter: 0,
                };
                self.ask_back(st)
            }
            LineSearch::Goldstein { c, beta, maxiter } => {
                if maxiter == 0 {
                    return self.finish(self.x0.clone(), f, 0.0);
                }
                let alpha = self.istep.max(1e-16);
                let st = Back {
                    f0: f,
                    slope,
                    alpha,
                    best_x: self.x0.clone(),
                    best_f: f,
                    best_alpha: 0.0,
                    c,
                    beta,
                    maxiter,
                    goldstein: true,
                    iter: 0,
                };
                self.ask_back(st)
            }
            LineSearch::Wolfe { c1, c2, maxiter } => {
                if !slope.is_finite() || slope >= 0.0 || maxiter == 0 {
                    return self.finish(self.x0.clone(), f, 0.0);
                }
                let alpha = self.istep.abs().max(1e-16);
                let st = Wolfe {
                    f0: f,
                    dphi0: slope,
                    c1,
                    c2,
                    maxiter,
                    alpha_max: 64.0_f64.max(self.istep.abs() * 64.0),
                    alpha_prev: 0.0,
                    phi_prev: f,
                    alpha,
                    i: 0,
                    best_x: self.x0.clone(),
                    best_f: f,
                    best_a: 0.0,
                    at: WolfeAt::Outer,
                    lo: 0.0,
                    hi: 0.0,
                    phi_lo: f,
                    zoom_iter: 0,
                    t: 0.0,
                    ft: f,
                };
                self.ask_wolfe(st, alpha)
            }
            LineSearch::Brent { maxiter, tol } => {
                let b = self.istep.max(1e-12);
                let st = Bracket {
                    f0: f,
                    a: 0.0,
                    b,
                    fa: 0.0,
                    fb: 0.0,
                    c: 0.0,
                    fc: 0.0,
                    u: 0.0,
                    it: 0,
                    maxiter,
                    tol,
                    stage: 0,
                };
                self.ask_bracket(st, 0.0)
            }
        }
    }

    fn back_offer(&mut self, mut st: Back, f: f64) -> SearchStep {
        let x = self.point(st.alpha);
        let accept = if st.goldstein {
            goldstein(f, st.f0, st.alpha, st.slope, st.c)
        } else {
            armijo(f, st.f0, st.alpha, st.slope, st.c)
        };
        if accept {
            return self.finish(x, f, st.alpha.abs());
        }
        if f < st.best_f {
            st.best_f = f;
            st.best_x = x;
            st.best_alpha = st.alpha;
        }
        if st.goldstein && armijo(f, st.f0, st.alpha, st.slope, st.c) {
            st.alpha /= st.beta;
        } else {
            st.alpha *= st.beta;
        }
        st.iter += 1;
        let alpha_max = 64.0_f64.max(self.istep.abs() * 64.0);
        if st.iter >= st.maxiter
            || !st.alpha.is_finite()
            || st.alpha < 1e-16
            || st.alpha > alpha_max
        {
            return self.best(st.best_x, st.best_f, st.best_alpha, st.f0);
        }
        let alpha = st.alpha;
        self.ask_back(st);
        self.evaluate(alpha)
    }

    fn wolfe_offer(&mut self, mut st: Wolfe, phi: f64, dphi: f64) -> SearchStep {
        match st.at {
            WolfeAt::Outer => self.wolfe_outer(st, phi, dphi),
            WolfeAt::ZoomLo => {
                st.phi_lo = phi;
                self.zoom_loop(st)
            }
            WolfeAt::ZoomTrial => self.zoom_trial(st, phi, dphi),
            WolfeAt::FinishValue => {
                let t = st.t;
                self.consider_tied(st, t, phi)
            }
            WolfeAt::FinishSlope => {
                let tied = strong_curvature(dphi, st.dphi0, st.c2);
                let t = st.t;
                let ft = st.ft;
                self.accept_finish(st, t, ft, tied)
            }
        }
    }

    fn wolfe_outer(&mut self, mut st: Wolfe, phi: f64, dphi: f64) -> SearchStep {
        let x = self.point(st.alpha);
        if phi.is_finite() && phi < st.best_f {
            st.best_f = phi;
            st.best_x = x.clone();
            st.best_a = st.alpha;
        }
        let fail_armijo = !phi.is_finite() || !armijo(phi, st.f0, st.alpha, st.dphi0, st.c1);
        if !fail_armijo && strong_curvature(dphi, st.dphi0, st.c2) {
            return self.finish(x, phi, st.alpha.abs());
        }
        if fail_armijo || (st.i > 0 && phi >= st.phi_prev) {
            let lo = st.alpha_prev;
            let hi = st.alpha;
            return self.enter_zoom(st, lo, hi);
        }
        if dphi >= 0.0 {
            let lo = st.alpha;
            let hi = st.alpha_prev;
            return self.enter_zoom(st, lo, hi);
        }
        if (st.alpha - st.alpha_max).abs() < 1e-16 {
            return self.best(st.best_x, st.best_f, st.best_a, st.f0);
        }
        st.alpha_prev = st.alpha;
        st.phi_prev = phi;
        st.alpha = (2.0 * st.alpha).min(st.alpha_max);
        st.i += 1;
        if st.i >= st.maxiter {
            return self.best(st.best_x, st.best_f, st.best_a, st.f0);
        }
        let alpha = st.alpha;
        self.ask_wolfe(st, alpha)
    }

    fn enter_zoom(&mut self, mut st: Wolfe, lo: f64, hi: f64) -> SearchStep {
        st.lo = lo;
        st.hi = hi;
        st.zoom_iter = 0;
        if lo == 0.0 {
            st.phi_lo = st.f0;
            self.zoom_loop(st)
        } else {
            st.at = WolfeAt::ZoomLo;
            self.ask_wolfe(st, lo)
        }
    }

    fn zoom_loop(&mut self, mut st: Wolfe) -> SearchStep {
        if !st.lo.is_finite()
            || !st.hi.is_finite()
            || (st.hi - st.lo).abs() < 1e-16
            || st.zoom_iter >= st.maxiter
        {
            let t = st.lo;
            let ft = st.phi_lo;
            return self.finish_zoom(st, t, ft);
        }
        let alpha = 0.5 * (st.lo + st.hi);
        st.zoom_iter += 1;
        st.at = WolfeAt::ZoomTrial;
        st.alpha = alpha;
        self.ask_wolfe(st, alpha)
    }

    fn zoom_trial(&mut self, mut st: Wolfe, phi_a: f64, dphi_a: f64) -> SearchStep {
        let alpha = st.alpha;
        if !phi_a.is_finite() {
            st.hi = alpha;
            return self.zoom_loop(st);
        }
        if armijo(phi_a, st.f0, alpha, st.dphi0, st.c1) && strong_curvature(dphi_a, st.dphi0, st.c2)
        {
            return self.finish_zoom(st, alpha, phi_a);
        }
        if !armijo(phi_a, st.f0, alpha, st.dphi0, st.c1) || phi_a >= st.phi_lo {
            st.hi = alpha;
        } else {
            if dphi_a * (st.hi - st.lo) >= 0.0 {
                st.hi = st.lo;
            }
            st.lo = alpha;
            st.phi_lo = phi_a;
        }
        self.zoom_loop(st)
    }

    fn finish_zoom(&mut self, mut st: Wolfe, t: f64, ft: f64) -> SearchStep {
        st.t = t;
        st.ft = ft;
        if t.is_finite() && t.abs() > 0.0 && !ft.is_finite() {
            st.at = WolfeAt::FinishValue;
            return self.ask_wolfe(st, t);
        }
        self.consider_tied(st, t, ft)
    }

    fn consider_tied(&mut self, mut st: Wolfe, t: f64, ft: f64) -> SearchStep {
        if t.is_finite() && t.abs() > 0.0 && ft == st.f0 && armijo(ft, st.f0, t, st.dphi0, st.c1) {
            st.t = t;
            st.ft = ft;
            st.at = WolfeAt::FinishSlope;
            return self.ask_wolfe(st, t);
        }
        self.accept_finish(st, t, ft, false)
    }

    fn accept_finish(&mut self, st: Wolfe, t: f64, ft: f64, tied: bool) -> SearchStep {
        if t.is_finite() && t.abs() > 0.0 && ft.is_finite() && (ft < st.f0 || tied) {
            self.finish(self.point(t), ft, t.abs())
        } else {
            self.best(st.best_x, st.best_f, st.best_a, st.f0)
        }
    }

    fn bracket_offer(&mut self, mut st: Bracket, f: f64) -> SearchStep {
        match st.stage {
            0 => {
                st.fa = f;
                st.stage = 1;
                let b = st.b;
                self.ask_bracket(st, b)
            }
            1 => {
                st.fb = f;
                if st.fa < st.fb {
                    std::mem::swap(&mut st.a, &mut st.b);
                    std::mem::swap(&mut st.fa, &mut st.fb);
                }
                st.c = st.b + GOLD * (st.b - st.a);
                st.stage = 2;
                let c = st.c;
                self.ask_bracket(st, c)
            }
            2 => {
                st.fc = f;
                st.it = 0;
                self.bracket_while(st)
            }
            _ => {
                st.a = st.b;
                st.b = st.c;
                st.fb = st.fc;
                st.c = st.u;
                st.fc = f;
                st.it += 1;
                st.fa = st.fb;
                self.bracket_while(st)
            }
        }
    }

    fn bracket_while(&mut self, mut st: Bracket) -> SearchStep {
        if st.fb >= st.fc && st.it < st.maxiter {
            st.u = st.b + GOLD * (st.c - st.b);
            st.stage = 3;
            let u = st.u;
            return self.ask_bracket(st, u);
        }
        let (lo, hi) = if st.a < st.c {
            (st.a, st.c)
        } else {
            (st.c, st.a)
        };
        let a = lo.min(hi);
        let b = lo.max(hi);
        let x = 0.5 * (a + b);
        let br = Brent {
            f0: st.f0,
            a,
            b,
            x,
            w: x,
            v: x,
            fx: 0.0,
            fw: 0.0,
            fv: 0.0,
            e: 0.0,
            d: 0.0,
            u: 0.0,
            iter: 0,
            maxiter: st.maxiter.max(20),
            tol: st.tol,
            init: true,
        };
        self.ask_brent(br, x)
    }

    fn brent_offer(&mut self, mut st: Brent, f: f64) -> SearchStep {
        if st.init {
            st.fx = f;
            st.fw = f;
            st.fv = f;
            st.init = false;
            return self.brent_continue(st);
        }
        let u = st.u;
        let fu = f;
        if fu <= st.fx {
            if u >= st.x {
                st.a = st.x;
            } else {
                st.b = st.x;
            }
            st.v = st.w;
            st.fv = st.fw;
            st.w = st.x;
            st.fw = st.fx;
            st.x = u;
            st.fx = fu;
        } else {
            if u < st.x {
                st.a = u;
            } else {
                st.b = u;
            }
            if fu <= st.fw || st.w == st.x {
                st.v = st.w;
                st.fv = st.fw;
                st.w = u;
                st.fw = fu;
            } else if fu <= st.fv || st.v == st.x || st.v == st.w {
                st.v = u;
                st.fv = fu;
            }
        }
        st.iter += 1;
        self.brent_continue(st)
    }

    fn brent_continue(&mut self, mut st: Brent) -> SearchStep {
        if st.iter >= st.maxiter {
            return self.finish_brent(st);
        }
        let xm = 0.5 * (st.a + st.b);
        let tol1 = st.tol * st.x.abs() + 1e-12;
        let tol2 = 2.0 * tol1;
        if (st.x - xm).abs() <= tol2 - 0.5 * (st.b - st.a) {
            return self.finish_brent(st);
        }
        let u = brent_u(&mut st, xm, tol1, tol2);
        st.u = u;
        self.ask_brent(st, u)
    }

    fn finish_brent(&mut self, st: Brent) -> SearchStep {
        if st.fx < st.f0 {
            self.finish(self.point(st.x), st.fx, st.x.abs())
        } else {
            self.finish(self.x0.clone(), st.f0, 0.0)
        }
    }

    fn ask_back(&mut self, st: Back) -> SearchStep {
        let alpha = st.alpha;
        self.state = State::Back(st);
        self.evaluate(alpha)
    }

    fn ask_wolfe(&mut self, st: Wolfe, alpha: f64) -> SearchStep {
        self.state = State::Wolfe(st);
        self.evaluate(alpha)
    }

    fn ask_bracket(&mut self, st: Bracket, alpha: f64) -> SearchStep {
        self.state = State::Bracket(st);
        self.evaluate(alpha)
    }

    fn ask_brent(&mut self, st: Brent, alpha: f64) -> SearchStep {
        self.state = State::Brent(st);
        self.evaluate(alpha)
    }

    fn best(&mut self, x: Array1<f64>, f: f64, alpha: f64, f0: f64) -> SearchStep {
        if f < f0 {
            self.finish(x, f, alpha.abs())
        } else {
            self.finish(self.x0.clone(), f0, 0.0)
        }
    }

    fn finish(&mut self, x: Array1<f64>, value: f64, alpha: f64) -> SearchStep {
        self.state = State::Done {
            x: x.clone(),
            value,
            alpha,
        };
        SearchStep::Accepted { x, value, alpha }
    }

    fn accepted(&self) -> SearchStep {
        match &self.state {
            State::Done { x, value, alpha } => SearchStep::Accepted {
                x: x.clone(),
                value: *value,
                alpha: *alpha,
            },
            _ => SearchStep::Accepted {
                x: self.x0.clone(),
                value: f64::INFINITY,
                alpha: 0.0,
            },
        }
    }

    fn evaluate(&self, alpha: f64) -> SearchStep {
        SearchStep::Evaluate {
            alpha,
            x: self.point(alpha),
        }
    }

    fn point(&self, t: f64) -> Array1<f64> {
        axpy(self.x0.view(), t, self.dir.view())
    }

    fn scored(&self, f: f64, g: ArrayView1<'_, f64>) -> f64 {
        if g.len() != self.dir.len() {
            f64::INFINITY
        } else {
            scored(f, g)
        }
    }

    fn directional(&self, g: ArrayView1<'_, f64>) -> f64 {
        if g.len() != self.dir.len() {
            return f64::NAN;
        }
        g.iter().zip(self.dir.iter()).map(|(gi, d)| gi * d).sum()
    }
}

/// The `u` Brent probes next. Updates `e` and `d` the way the in-library
/// search does before it calls `φ`.
fn brent_u(st: &mut Brent, xm: f64, tol1: f64, tol2: f64) -> f64 {
    let mut u;
    if st.e.abs() > tol1 {
        let r = (st.x - st.w) * (st.fx - st.fv);
        let mut q = (st.x - st.v) * (st.fx - st.fw);
        let mut p = (st.x - st.v) * q - (st.x - st.w) * r;
        q = 2.0 * (q - r);
        if q > 0.0 {
            p = -p;
        }
        q = q.abs();
        let etemp = st.e;
        st.e = st.d;
        if p.abs() >= (0.5 * q * etemp).abs() || p <= q * (st.a - st.x) || p >= q * (st.b - st.x) {
            st.e = if st.x >= xm { st.a - st.x } else { st.b - st.x };
            st.d = CGOLD * st.e;
        } else {
            st.d = p / q;
            u = st.x + st.d;
            if u - st.a < tol2 || st.b - u < tol2 {
                st.d = if xm - st.x >= 0.0 { tol1 } else { -tol1 };
            }
        }
    } else {
        st.e = if st.x >= xm { st.a - st.x } else { st.b - st.x };
        st.d = CGOLD * st.e;
    }
    u = if st.d.abs() >= tol1 {
        st.x + st.d
    } else {
        st.x + if st.d >= 0.0 { tol1 } else { -tol1 }
    };
    u
}
