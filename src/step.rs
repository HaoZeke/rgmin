//! Shared accept / clip / scale for one line-search move.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::linesearch::LineSearch;

pub(crate) fn l2(g: &Array1<f64>) -> f64 {
    crate::vecops::nrm2(g.view())
}

pub(crate) fn scale_step(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
    let mut n2 = 0.0;
    for i in 0..trial.len() {
        let d = trial[i] - origin[i];
        n2 += d * d;
    }
    let n = n2.sqrt();
    if n > cap && n > 0.0 {
        let s = cap / n;
        for i in 0..trial.len() {
            trial[i] = origin[i] + s * (trial[i] - origin[i]);
        }
    }
}

/// eOn `maxAtomMotionAppliedV`: scale the whole step so the largest
/// per-atom displacement is at most `cap`.
pub(crate) fn scale_step_atom(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
    let n = trial.len();
    let mut max_atom = 0.0;
    let mut i = 0;
    while i + 3 <= n {
        let dx = trial[i] - origin[i];
        let dy = trial[i + 1] - origin[i + 1];
        let dz = trial[i + 2] - origin[i + 2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r > max_atom {
            max_atom = r;
        }
        i += 3;
    }
    if i < n {
        let mut r2 = 0.0;
        while i < n {
            let d = trial[i] - origin[i];
            r2 += d * d;
            i += 1;
        }
        max_atom = max_atom.max(r2.sqrt());
    }
    if max_atom > cap && max_atom > 0.0 {
        let s = cap / max_atom;
        for k in 0..n {
            trial[k] = origin[k] + s * (trial[k] - origin[k]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn atom_cap_does_not_crush_a_uniform_cluster_step() {
        // 2 atoms each move 0.15. Per-atom cap 0.2 keeps the step.
        // A Euclidean 0.2 cap would scale it down.
        let origin = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut trial = array![0.15, 0.0, 0.0, 1.15, 0.0, 0.0];
        scale_step_atom(&origin, &mut trial, 0.2);
        assert!((trial[0] - 0.15).abs() < 1e-15);
        assert!((trial[3] - 1.15).abs() < 1e-15);
        let mut eucl = array![0.15, 0.0, 0.0, 1.15, 0.0, 0.0];
        scale_step(&origin, &mut eucl, 0.2);
        assert!(eucl[0] < 0.15 - 1e-6);
    }
}

/// Line search, clip to bounds, optional maxmove. Returns `(x, f, |α|, moved)`.
pub(crate) fn take_step<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    dir: ArrayView1<'_, f64>,
    istep: f64,
    linesearch: LineSearch,
    control: &Control,
) -> (Array1<f64>, f64, f64, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let (npos, _, lsstep) =
        linesearch.search(|x| obj.value_and_gradient(x), pos.view(), dir, istep);
    let mut trial = obj.bounds().clip(npos.view());
    if let Some(cap) = control.maxmove {
        scale_step(pos, &mut trial, cap);
    }
    let (nval, grad) = obj.value_and_gradient(trial.view());
    let finite = nval.is_finite() && grad.iter().all(|g| g.is_finite());
    // Bounds and maxmove change the point returned by the line search.
    // Acceptance uses that point's value and directional derivative. At
    // equal rounded energies, strong curvature certifies a descent step.
    let curvature_decrease = match linesearch {
        LineSearch::Wolfe { c1, c2, .. } if finite && nval == value => {
            let (_, initial_grad) = obj.value_and_gradient(pos.view());
            let delta = &trial - pos;
            let initial_slope = initial_grad.dot(&delta);
            let trial_slope = grad.dot(&delta);
            initial_slope < 0.0
                && crate::linesearch::conditions::armijo(nval, value, 1.0, initial_slope, c1)
                && crate::linesearch::conditions::strong_curvature(trial_slope, initial_slope, c2)
        }
        _ => false,
    };
    if finite && (nval < value || curvature_decrease) {
        (trial, nval, lsstep, true)
    } else {
        (pos.clone(), value, 0.0, false)
    }
}

pub(crate) fn next_istep(lsstep: f64, control: &Control) -> f64 {
    if lsstep <= 0.0 {
        control.istep
    } else {
        lsstep * 0.5
    }
}
