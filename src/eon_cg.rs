//! Line-search-free conjugate gradients, the eOn client step.
//!
//! `ConjugateGradients::single_step` at TheochemUI/eOn
//! `579738a9387e8d0ea55496f45dd057812cfab892`: Polak-Ribiere on the
//! force, restarted when the forces are not sufficiently orthogonal,
//! with the step length from one curvature probe. No line search.
//!
//! Hestenes and Stiefel, *Methods of conjugate gradients for solving
//! linear systems*, <https://doi.org/10.6028/jres.049.044>.
//! Polak and Ribiere, *Note sur la convergence de methodes de directions
//! conjuguees*, Revue Francaise d'Informatique et de Recherche
//! Operationnelle 16 (1969).
//!
//! The buffers live for the whole solve. A step does not allocate.

/// Controls for [`minimize_eon_cg`].
#[derive(Clone, Copy, Debug)]
pub struct EonCg {
    /// Longest per-atom displacement, eOn `max_move`.
    pub max_move: f64,
    /// Stop when the selected gradient norm is below this value.
    pub gtol: f64,
    /// Outer iterations.
    pub max_iter: usize,
    /// `true`: max atom force. `false`: infinity norm.
    pub cluster: bool,
}

impl Default for EonCg {
    fn default() -> Self {
        Self {
            max_move: crate::EON_MAX_MOVE,
            gtol: 1e-5,
            max_iter: 20_000,
            cluster: false,
        }
    }
}

/// Result of [`minimize_eon_cg`].
#[derive(Clone, Copy, Debug)]
pub struct EonCgReport {
    /// Energy at the returned point.
    pub energy: f64,
    /// Gradient norm in the norm selected by [`EonCg::cluster`].
    pub grad: f64,
    /// Force evaluations, including curvature probes.
    pub evals: usize,
    /// The gradient test was met.
    pub converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += *x * *y;
    }
    s
}

fn max_atom(v: &[f64]) -> f64 {
    let mut m = 0.0;
    let mut i = 0;
    while i + 3 <= v.len() {
        let r = (v[i] * v[i] + v[i + 1] * v[i + 1] + v[i + 2] * v[i + 2]).sqrt();
        if r > m {
            m = r;
        }
        i += 3;
    }
    if i < v.len() {
        let mut r2 = 0.0;
        while i < v.len() {
            r2 += v[i] * v[i];
            i += 1;
        }
        m = m.max(r2.sqrt());
    }
    m
}

fn grad_norm(g: &[f64], cluster: bool) -> f64 {
    if !cluster {
        let mut m = 0.0_f64;
        for v in g {
            m = m.max(v.abs());
        }
        return m;
    }
    max_atom(g)
}

/// Minimize with the eOn conjugate-gradient step.
///
/// `fill` writes the gradient of the energy it returns into `g`.
/// The solve allocates its workspace once.
pub fn minimize_eon_cg<F>(x: &mut [f64], mut fill: F, opts: &EonCg) -> EonCgReport
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut force = vec![0.0; n];
    let mut force_old = vec![0.0; n];
    let mut direction = vec![0.0; n];
    let mut direction_old = vec![0.0; n];
    let mut probe = vec![0.0; n];
    let mut g_probe = vec![0.0; n];
    let mut energy = fill(x, &mut g);
    let mut evals = 1usize;
    let mut converged = false;
    let fd = crate::EON_FINITE_DIFFERENCE;
    for _ in 0..opts.max_iter {
        let gnorm = grad_norm(&g, opts.cluster);
        if gnorm < opts.gtol {
            converged = true;
            break;
        }
        for i in 0..n {
            force[i] = -g[i];
        }
        let b = dot(&force_old, &force_old);
        let cross = dot(&force, &force_old).abs();
        let mut gamma = 0.0;
        if b > 0.0 && cross < 0.5 * b {
            let mut num = 0.0;
            for i in 0..n {
                num += force[i] * (force[i] - force_old[i]);
            }
            gamma = num / b;
        }
        for i in 0..n {
            direction[i] = force[i] + gamma * direction_old[i];
        }
        direction_old.copy_from_slice(&direction);
        force_old.copy_from_slice(&force);
        let dn = dot(&direction, &direction).sqrt();
        let inv = if dn > 0.0 { 1.0 / dn } else { 0.0 };
        for i in 0..n {
            probe[i] = x[i] + fd * direction[i] * inv;
        }
        let _ = fill(&probe, &mut g_probe);
        evals += 1;
        let mut p1 = 0.0;
        let mut p2 = 0.0;
        for i in 0..n {
            let u = direction[i] * inv;
            p1 += force[i] * u;
            p2 += -g_probe[i] * u;
        }
        let curv = (p1 - p2) / fd;
        let mut step = opts.max_move;
        if curv > 0.0 && curv.is_finite() {
            step = p1 / curv;
        }
        if step.abs() > opts.max_move {
            step = step.copysign(opts.max_move);
        }
        for d in &mut direction {
            *d *= inv * step;
        }
        let motion = max_atom(&direction);
        if motion > opts.max_move && motion > 0.0 {
            let s = opts.max_move / motion;
            for v in &mut direction {
                *v *= s;
            }
        }
        for i in 0..n {
            x[i] += direction[i];
        }
        energy = fill(x, &mut g);
        evals += 1;
        if step >= opts.max_move {
            direction_old.fill(0.0);
            force_old.fill(0.0);
        }
    }
    EonCgReport {
        energy,
        grad: grad_norm(&g, opts.cluster),
        evals,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosen(x: &[f64], g: &mut [f64]) -> f64 {
        g.fill(0.0);
        let mut f = 0.0;
        for i in 0..x.len() - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i] * x[i];
            f += a * a + 100.0 * b * b;
            g[i] += -2.0 * a - 400.0 * x[i] * b;
            g[i + 1] += 200.0 * b;
        }
        f
    }

    #[test]
    fn rosenbrock_2_matches_the_eon_evaluation_count() {
        let mut x = [-1.2, 1.0];
        let opts = EonCg {
            gtol: 1e-5,
            cluster: false,
            ..EonCg::default()
        };
        let rep = minimize_eon_cg(&mut x, rosen, &opts);
        assert!(rep.converged, "{rep:?} x {x:?}");
        assert_eq!(rep.evals, 87, "evals {}", rep.evals);
        assert!(rep.energy < 1e-8, "{}", rep.energy);
    }
}
