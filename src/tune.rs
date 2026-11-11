//! Training-set choice of L-BFGS memory and cap.
//!
//! The selector only sees rows the caller labels as training. A setting
//! is eligible when every training problem converged. The winner is the
//! eligible setting with the smallest total force-call count. Ties break
//! toward cap `0.2`, then toward memory `20`, the eOn client defaults.

/// One training run of one `(memory, cap)` setting.
#[derive(Clone, Debug)]
pub struct TrainRow {
    /// Correction pairs.
    pub memory: usize,
    /// Per-atom cap.
    pub cap: f64,
    /// Force calls, probes included.
    pub evals: usize,
    /// The gradient test was met inside the budget.
    pub converged: bool,
    /// Problem name. A setting must converge on every name in the slice.
    pub problem: String,
}

/// `(memory, cap)` with the smallest total training force calls.
///
/// `None` when no setting converges on every problem, or the slice is
/// empty. A non-finite cap is ignored. Ties break toward cap `0.2`,
/// then toward memory `20`.
#[must_use]
pub fn select_train_setting(rows: &[TrainRow]) -> Option<(usize, f64)> {
    use std::collections::BTreeMap;
    let mut problems = BTreeMap::<String, ()>::new();
    for row in rows {
        problems.insert(row.problem.clone(), ());
    }
    if problems.is_empty() {
        return None;
    }
    let need = problems.len();
    let mut grouped: BTreeMap<(u64, usize), Vec<&TrainRow>> = BTreeMap::new();
    for row in rows {
        if !row.cap.is_finite() || row.cap <= 0.0 {
            continue;
        }
        grouped
            .entry((row.cap.to_bits(), row.memory))
            .or_default()
            .push(row);
    }
    let mut best: Option<(usize, u64, usize, f64)> = None;
    for group in grouped.values() {
        let mut seen = BTreeMap::<String, usize>::new();
        let mut ok = true;
        for row in group {
            if !row.converged {
                ok = false;
                break;
            }
            *seen.entry(row.problem.clone()).or_insert(0) += row.evals;
        }
        if !ok || seen.len() != need {
            continue;
        }
        let total: usize = seen.values().copied().sum();
        let memory = group[0].memory;
        let cap = group[0].cap;
        let key = (
            total,
            ((cap - 0.2).abs() * 1.0e12).round() as u64,
            memory.abs_diff(20),
        );
        let take = match best {
            None => true,
            Some((bt, _, bm, bc)) => {
                key < (
                    bt,
                    ((bc - 0.2).abs() * 1.0e12).round() as u64,
                    bm.abs_diff(20),
                )
            }
        };
        if take {
            best = Some((total, key.1, memory, cap));
        }
    }
    best.map(|(_, _, memory, cap)| (memory, cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(memory: usize, cap: f64, evals: usize, converged: bool, problem: &str) -> TrainRow {
        TrainRow {
            memory,
            cap,
            evals,
            converged,
            problem: problem.to_string(),
        }
    }

    #[test]
    fn the_selector_drops_a_setting_that_misses_one_problem() {
        let rows = vec![
            row(8, 0.15, 10, true, "a"),
            row(8, 0.15, 12, true, "b"),
            row(8, 0.15, 40, false, "c"),
            row(20, 0.2, 30, true, "a"),
            row(20, 0.2, 30, true, "b"),
            row(20, 0.2, 30, true, "c"),
        ];
        let (memory, cap) = select_train_setting(&rows).unwrap();
        assert_eq!(memory, 20);
        assert!((cap - 0.2).abs() < 1e-15);
    }

    #[test]
    fn a_lower_total_wins_when_every_problem_converges() {
        let rows = vec![
            row(20, 0.2, 50, true, "a"),
            row(20, 0.2, 50, true, "b"),
            row(12, 0.25, 20, true, "a"),
            row(12, 0.25, 20, true, "b"),
        ];
        let (memory, cap) = select_train_setting(&rows).unwrap();
        assert_eq!(memory, 12);
        assert!((cap - 0.25).abs() < 1e-15);
    }

    #[test]
    fn nothing_is_selected_when_no_setting_covers_the_set() {
        let rows = vec![
            row(8, 0.2, 10, true, "a"),
            row(8, 0.2, 10, false, "b"),
            row(20, 0.2, 10, false, "a"),
            row(20, 0.2, 10, true, "b"),
        ];
        assert!(select_train_setting(&rows).is_none());
    }
}
