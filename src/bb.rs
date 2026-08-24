//! Barzilai-Borwein two-point spectral step.
//!
//! Barzilai and Borwein, *Two-Point Step Size Gradient Methods*,
//! <https://doi.org/10.1093/imanum/8.1.141>.
//! Raydan, *The Barzilai and Borwein Gradient Method for the Large
//! Scale Unconstrained Minimization Problem*,
//! <https://doi.org/10.1137/S1052623494266365>.
//! Iannazzo and Porcelli, *The Riemannian Barzilai-Borwein method
//! with nonmonotone line-search and the matrix geometric mean
//! computation*, <https://doi.org/10.1093/imanum/drx015>.
//! manopt `barzilaiborwein`: `rgrad = egrad2rgrad`, then
//! `s = T(step)`, `y = g+ - T(g)` at the arrival tangent.

use ndarray::Array1;

use crate::vecops::{self, Vector};

const CURVATURE: f64 = 1e-12;
const ALPHA_MIN: f64 = 1e-12;
const ALPHA_MAX: f64 = 1e12;

/// BB1 length `α = (s·s)/(s·y)`. Falls back to `istep` when `s·y`
/// is not safely positive. manopt `strategy = 'direct'`.
pub fn bb1_alpha(s: &Array1<f64>, y: &Array1<f64>, istep: f64) -> f64 {
    let sy = vecops::dot(s.view(), y.view());
    if sy <= CURVATURE {
        return istep.max(ALPHA_MIN);
    }
    let ss = vecops::dot(s.view(), s.view());
    (ss / sy).clamp(ALPHA_MIN, ALPHA_MAX)
}

/// Spectral descent direction `-α g`.
pub fn bb_direction(
    s: Option<&Array1<f64>>,
    y: Option<&Array1<f64>>,
    g: &Array1<f64>,
    istep: f64,
) -> Array1<f64> {
    let alpha = match (s, y) {
        (Some(s), Some(y)) => bb1_alpha(s, y, istep),
        _ => istep.max(ALPHA_MIN),
    };
    let g = Vector::from_host(g.clone());
    let mut dir = Vector::zeros_cpu(g.host_view().len());
    vecops::vaxpy(-alpha, &g, &mut dir);
    dir.into_host()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bb1_recovers_the_quadratic_scale() {
        // f = (1/2) x^T diag(2, 8) x, so H = diag(2, 8).
        // A step s = (1, 1) gives y = H s = (2, 8), α = 2/10 = 0.2.
        let s = array![1.0, 1.0];
        let y = array![2.0, 8.0];
        let a = bb1_alpha(&s, &y, 1.0);
        assert!((a - 0.2).abs() < 1e-15);
    }

    #[test]
    fn negative_curvature_falls_back() {
        let s = array![1.0];
        let y = array![-1.0];
        assert!((bb1_alpha(&s, &y, 0.3) - 0.3).abs() < 1e-15);
    }

    #[test]
    fn direction_is_minus_alpha_g() {
        let g = array![1.0, 0.0];
        let d = bb_direction(None, None, &g, 0.5);
        assert!((d[0] + 0.5).abs() < 1e-15);
        assert!(d[1].abs() < 1e-15);
        let s = array![1.0, 1.0];
        let y = array![2.0, 8.0];
        let d = bb_direction(Some(&s), Some(&y), &g, 1.0);
        assert!((d[0] + 0.2).abs() < 1e-15);
        assert!(d[1].abs() < 1e-15);
    }
}
