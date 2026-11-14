//! Unlimited L-BFGS plateau.
//!
//! A caller's own step loop with a zero iteration budget otherwise never
//! returns when accepted steps stop making progress: the coordinates
//! still move, `||g||` stays above `gtol`, and neither the value nor the
//! gradient norm changes by more than the objective can resolve. One such
//! step can be an overshoot the next step corrects, which is the whole
//! trajectory of a 1-D problem, so the run is [`period`] long: `max(n, 2)`
//! accepted steps. A positive budget does not consult this.
//!
//! The value is held to the session's objective roundoff window, the same
//! window that let the line search accept a step without a measured
//! decrease. The gradient norm is held to `sqrt(eps)` relative, the SCG
//! plateau band. Both compare against the step that opened the run, so a
//! slow descent that is measurable over the run opens a new one.

/// Relative band on `||g||` inside which a step is the same plateau.
const GRAD_BAND: f64 = 1.490_116_119_384_765_6e-8;

/// Consecutive accepted steps whose value and gradient norm stay within
/// the resolution of the step that opened the run.
#[derive(Clone, Debug, Default)]
pub(crate) struct RepeatedAcceptedStep {
    value: f64,
    grad_norm: f64,
    /// Accepted steps in the open run, counting the one that opened it.
    run: usize,
    open: bool,
}

impl RepeatedAcceptedStep {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record one accepted step. `moved` is false when the coordinates did
    /// not change: that step is not this plateau. `window` is the relative
    /// objective roundoff. Returns whether the run has reached [`period`]
    /// with `||g||` still at or above `gtol`.
    pub(crate) fn observe(
        &mut self,
        dim: usize,
        moved: bool,
        value: f64,
        grad_norm: f64,
        gtol: f64,
        window: f64,
    ) -> bool {
        let flat = moved && value.is_finite() && grad_norm.is_finite() && grad_norm >= gtol;
        let continues = self.open
            && flat
            && (value - self.value).abs() <= window * self.value.abs()
            && (grad_norm - self.grad_norm).abs() <= GRAD_BAND * self.grad_norm;
        if continues {
            self.run += 1;
        } else {
            self.value = value;
            self.grad_norm = grad_norm;
            self.run = 1;
            self.open = flat;
        }
        continues && self.run >= period(dim)
    }
}

/// Accepted steps that repeat before an unlimited L-BFGS fit stops.
///
/// `n` alone stops a 1-D fit on its first flat step, and that step is an
/// overshoot. Two is the shortest run that can be told from one.
pub(crate) fn period(dim: usize) -> usize {
    dim.max(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS4: f64 = 4.0 * f64::EPSILON;

    fn flat(
        mut seen: RepeatedAcceptedStep,
        dim: usize,
        steps: usize,
    ) -> (RepeatedAcceptedStep, usize) {
        let mut stopped = 0;
        for _ in 0..steps {
            if seen.observe(dim, true, 759_036.0, 2.14983, 1e-2, EPS4) {
                stopped += 1;
            }
        }
        (seen, stopped)
    }

    #[test]
    fn one_dimension_waits_two_steps_and_not_one() {
        let (seen, stopped) = flat(RepeatedAcceptedStep::new(), 1, 1);
        assert_eq!(stopped, 0);
        assert_eq!(period(1), 2);
        let (_, stopped) = flat(seen, 1, 1);
        assert_eq!(stopped, 1);
    }

    #[test]
    fn wider_problem_waits_its_dimension() {
        assert_eq!(period(4), 4);
        let (seen, stopped) = flat(RepeatedAcceptedStep::new(), 4, 3);
        assert_eq!(stopped, 0);
        assert_eq!(seen.run, 3);
        let (_, stopped) = flat(seen, 4, 1);
        assert_eq!(stopped, 1);
    }

    #[test]
    fn a_step_that_changes_the_gradient_opens_a_new_run() {
        let mut seen = RepeatedAcceptedStep::new();
        assert!(!seen.observe(4, true, 1.0, 2.0, 1e-2, EPS4));
        assert!(!seen.observe(4, true, 1.0, 2.0, 1e-2, EPS4));
        assert!(!seen.observe(4, true, 1.0, 1.5, 1e-2, EPS4));
        assert_eq!(seen.run, 1);
    }

    #[test]
    fn a_step_below_the_gradient_tolerance_is_not_the_plateau() {
        let mut seen = RepeatedAcceptedStep::new();
        assert!(!seen.observe(3, true, 1.0, 1e-8, 1e-2, EPS4));
        assert!(!seen.observe(3, true, 1.0, 1e-8, 1e-2, EPS4));
        assert!(!seen.observe(3, true, 1.0, 1e-8, 1e-2, EPS4));
        assert_eq!(seen.run, 1);
    }

    #[test]
    fn the_band_is_relative_to_the_step_that_opened_the_run() {
        assert_eq!(GRAD_BAND, f64::EPSILON.sqrt());
        let f = 759_036.0;
        let ulp = f64::from_bits(f64::to_bits(f) + 1) - f;
        let mut seen = RepeatedAcceptedStep::new();
        assert!(!seen.observe(2, true, f, 2.0, 1e-2, EPS4));
        assert!(seen.observe(2, true, f + 2.0 * ulp, 2.0 * (1.0 + 1e-12), 1e-2, EPS4));
        // Each step inside the band of its predecessor, but the run as a
        // whole a measurable descent: the opener resets it.
        let mut seen = RepeatedAcceptedStep::new();
        let mut value = f;
        for _ in 0..8 {
            assert!(!seen.observe(8, true, value, 2.0, 1e-2, EPS4));
            value -= 3.0 * ulp;
        }
    }

    #[test]
    fn a_falling_gradient_norm_is_progress() {
        let mut seen = RepeatedAcceptedStep::new();
        let mut g = 2.0;
        for _ in 0..8 {
            assert!(!seen.observe(2, true, 1.0, g, 1e-2, EPS4));
            g *= 1.0 - 1e-6;
        }
    }

    #[test]
    fn a_wider_window_admits_a_noisier_value_and_zero_admits_only_equality() {
        let mut seen = RepeatedAcceptedStep::new();
        assert!(!seen.observe(2, true, 1.0, 2.0, 1e-2, 1e-6));
        assert!(seen.observe(2, true, 1.0 + 5e-7, 2.0, 1e-2, 1e-6));
        let mut seen = RepeatedAcceptedStep::new();
        assert!(!seen.observe(2, true, 1.0, 2.0, 1e-2, 0.0));
        assert!(!seen.observe(2, true, 1.0 + f64::EPSILON, 2.0, 1e-2, 0.0));
        assert!(seen.observe(2, true, 1.0 + f64::EPSILON, 2.0, 1e-2, 0.0));
    }

    #[test]
    fn a_step_that_does_not_move_or_is_not_finite_is_not_the_plateau() {
        let mut seen = RepeatedAcceptedStep::new();
        assert!(!seen.observe(2, true, 1.0, 2.0, 1e-2, EPS4));
        assert!(!seen.observe(2, false, 1.0, 2.0, 1e-2, EPS4));
        assert!(!seen.observe(2, true, 1.0, 2.0, 1e-2, EPS4));
        assert!(!seen.observe(2, true, f64::NAN, 2.0, 1e-2, EPS4));
        assert!(!seen.observe(2, true, 1.0, f64::INFINITY, 1e-2, EPS4));
    }
}
