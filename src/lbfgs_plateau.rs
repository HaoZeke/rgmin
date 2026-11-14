//! Unlimited L-BFGS plateau.
//!
//! The decision lives here, not in the line search or the energy
//! acceptance, so a change to the roundoff window does not meet it.
//!
//! A zero iteration budget otherwise never returns when an accepted step
//! repeats: the value and the gradient norm stay put, the coordinates
//! still move, and `||g||` stays above `gtol`. One such step can be an
//! overshoot the next step corrects, which is the whole trajectory of a
//! 1-D problem, so the run is [`period`] long: `max(n, 2)` accepted steps.
//! A positive budget does not consult this.

/// Consecutive accepted steps whose value and gradient norm are the same
/// bits as the step that opened the run.
#[derive(Clone, Debug, Default)]
pub(crate) struct RepeatedAcceptedStep {
    value_bits: u64,
    grad_bits: u64,
    /// Accepted steps in the open run, counting the one that opened it.
    run: usize,
    open: bool,
}

impl RepeatedAcceptedStep {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record one accepted step. `moved` is false when the coordinates did
    /// not change: that step is not this plateau. Returns whether the run
    /// has reached [`period`] with `||g||` still at or above `gtol`.
    pub(crate) fn observe(
        &mut self,
        dim: usize,
        moved: bool,
        value: f64,
        grad_norm: f64,
        gtol: f64,
    ) -> bool {
        let value_bits = value.to_bits();
        let grad_bits = grad_norm.to_bits();
        let continues = self.open
            && moved
            && value_bits == self.value_bits
            && grad_bits == self.grad_bits
            && grad_norm >= gtol;
        if continues {
            self.run += 1;
        } else {
            self.value_bits = value_bits;
            self.grad_bits = grad_bits;
            self.run = 1;
            self.open = true;
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

    fn flat(
        mut seen: RepeatedAcceptedStep,
        dim: usize,
        steps: usize,
    ) -> (RepeatedAcceptedStep, usize) {
        let mut stopped = 0;
        for _ in 0..steps {
            if seen.observe(dim, true, 759_036.0, 2.14983, 1e-2) {
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
        assert!(!seen.observe(4, true, 1.0, 2.0, 1e-2));
        assert!(!seen.observe(4, true, 1.0, 2.0, 1e-2));
        assert!(!seen.observe(4, true, 1.0, 1.5, 1e-2));
        assert_eq!(seen.run, 1);
    }

    #[test]
    fn a_step_below_the_gradient_tolerance_is_not_the_plateau() {
        let mut seen = RepeatedAcceptedStep::new();
        assert!(!seen.observe(3, true, 1.0, 1e-8, 1e-2));
        assert!(!seen.observe(3, true, 1.0, 1e-8, 1e-2));
        assert!(!seen.observe(3, true, 1.0, 1e-8, 1e-2));
        assert_eq!(seen.run, 1);
    }
}
