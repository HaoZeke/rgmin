//! Doubly-stochastic matrices (Birkhoff polytope) with the Fisher metric.
//!
//! manopt `multinomialdoublystochasticfactory`. The point is an n-by-n
//! matrix, packed row-major, with positive entries and unit row and
//! column sums. The tangent space is `{V : V 1 = 0, V^T 1 = 0}`. The
//! Fisher-orthogonal projection at `X` solves
//! `[I X; X^T I] [alpha; beta] = [V 1; V^T 1]` and returns
//! `V - X ⊙ (alpha 1^T + 1 beta^T)`. Retraction is
//! `X ⊙ exp(V ⊘ X)` followed by Sinkhorn. Transport is projection
//! at the arrival point.
//!
//! Distinct from [`super::Multinomial`] (one simplex) and from the
//! reserved grassmann / hyperbolic / SPD tokens 7-10.

use ndarray::{Array1, Array2, ArrayView1};

use crate::vecops;

use super::Manifold;

/// Birkhoff polytope, Fisher information metric.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialDoublyStochastic {
    /// Side length. Packed length is `n * n`.
    pub n: usize,
}

impl MultinomialDoublyStochastic {
    /// Row-major flatten of an n-by-n matrix (manopt `M.vec` with
    /// rgmin's row-major packing).
    pub fn pack(mat: &Array2<f64>) -> Array1<f64> {
        let (n, m) = mat.dim();
        let mut p = Array1::zeros(n * m);
        for i in 0..n {
            for j in 0..m {
                p[i * m + j] = mat[(i, j)];
            }
        }
        p
    }

    /// Inverse of [`Self::pack`] for this side length.
    pub fn unpack(&self, packed: &Array1<f64>) -> Array2<f64> {
        let n = self.n;
        let mut a = Array2::<f64>::zeros((n, n));
        if packed.len() != n * n {
            return a;
        }
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = packed[i * n + j];
            }
        }
        a
    }

    fn fits(&self, len: usize) -> bool {
        self.n >= 2 && len == self.n * self.n
    }

    fn row_col_sums(&self, a: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let n = self.n;
        let mut r = Array1::<f64>::zeros(n);
        let mut c = Array1::<f64>::zeros(n);
        for i in 0..n {
            r[i] = vecops::sum(a.row(i));
        }
        for j in 0..n {
            c[j] = vecops::sum(a.column(j));
        }
        (r, c)
    }

    /// Apply `[I X; X^T I]` to a stacked `[alpha; beta]`.
    fn apply_saddle(&self, x: &Array2<f64>, z: ArrayView1<f64>) -> Array1<f64> {
        let n = self.n;
        let mut out = Array1::<f64>::zeros(2 * n);
        let alpha = z.slice(ndarray::s![0..n]);
        let beta = z.slice(ndarray::s![n..]);
        for i in 0..n {
            out[i] = alpha[i] + vecops::dot(x.row(i), beta);
        }
        for j in 0..n {
            out[n + j] = beta[j] + vecops::dot(x.column(j), alpha);
        }
        out
    }

    /// Particular solution of the singular saddle
    /// `[I X; X^T I] [alpha; beta] = [row; col]`.
    fn duals(&self, x: &Array2<f64>, eta: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let n = self.n;
        let (r, c) = self.row_col_sums(eta);
        let mut b = Array1::<f64>::zeros(2 * n);
        for i in 0..n {
            b[i] = r[i];
            b[n + i] = c[i];
        }
        let mut z = Array1::<f64>::zeros(2 * n);
        let mut resid = b.clone();
        let mut p = resid.clone();
        let mut rsold = vecops::dot(resid.view(), resid.view());
        if rsold > 1e-30 {
            for _ in 0..(4 * n + 16).max(32) {
                let ap = self.apply_saddle(x, p.view());
                let denom = vecops::dot(p.view(), ap.view());
                if denom.abs() < 1e-30 {
                    break;
                }
                let step = rsold / denom;
                vecops::axpy(step, p.view(), &mut z);
                vecops::axpy(-step, ap.view(), &mut resid);
                let rsnew = vecops::dot(resid.view(), resid.view());
                if rsnew.sqrt() < 1e-12 {
                    break;
                }
                let beta = rsnew / rsold;
                for i in 0..2 * n {
                    p[i] = resid[i] + beta * p[i];
                }
                rsold = rsnew;
            }
        }
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            alpha[i] = z[i];
            beta[i] = z[n + i];
        }
        (alpha, beta)
    }

    fn fisher_proj(&self, x: &Array2<f64>, eta: &Array2<f64>) -> Array2<f64> {
        let n = self.n;
        let (alpha, beta) = self.duals(x, eta);
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] = eta[(i, j)] - (alpha[i] + beta[j]) * x[(i, j)];
            }
        }
        out
    }

    fn sinkhorn(&self, mut a: Array2<f64>) -> Array2<f64> {
        let n = self.n;
        let iters = 100 + 2 * n;
        for _ in 0..iters {
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = a[(i, j)].max(f64::EPSILON);
                }
                let rs = vecops::sum(a.row(i));
                if rs > 0.0 && rs.is_finite() {
                    for j in 0..n {
                        a[(i, j)] /= rs;
                    }
                }
            }
            for j in 0..n {
                let cs = vecops::sum(a.column(j));
                if cs > 0.0 && cs.is_finite() {
                    for i in 0..n {
                        a[(i, j)] /= cs;
                    }
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = a[(i, j)].max(f64::EPSILON);
            }
        }
        a
    }
}

impl Manifold for MultinomialDoublyStochastic {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.fits(n) {
            Ok(())
        } else {
            Err(self.n.max(2) * self.n.max(2))
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || !self.fits(v.len()) {
            return v.clone();
        }
        let xm = self.unpack(x);
        let vm = self.unpack(v);
        Self::pack(&self.fisher_proj(&xm, &vm))
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || !self.fits(egrad.len()) {
            return egrad.clone();
        }
        let xm = self.unpack(x);
        let gm = self.unpack(egrad);
        let mut mu = Array2::<f64>::zeros((self.n, self.n));
        for i in 0..self.n {
            for j in 0..self.n {
                mu[(i, j)] = xm[(i, j)] * gm[(i, j)];
            }
        }
        Self::pack(&self.fisher_proj(&xm, &mu))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || !self.fits(v.len()) {
            return x + v;
        }
        let xm = self.unpack(x);
        let vm = self.unpack(v);
        let mut y = Array2::<f64>::zeros((self.n, self.n));
        for i in 0..self.n {
            for j in 0..self.n {
                let xij = xm[(i, j)].max(f64::EPSILON);
                y[(i, j)] = xij * (vm[(i, j)] / xij).exp();
            }
        }
        Self::pack(&self.sinkhorn(y))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn barycenter(n: usize) -> Array1<f64> {
        Array1::from_elem(n * n, 1.0 / n as f64)
    }

    fn assert_doubly_stochastic(y: &Array1<f64>, n: usize, tol: f64) {
        assert_eq!(y.len(), n * n);
        assert!(y.iter().all(|&yi| yi > 0.0), "left the interior {y:?}");
        let m = MultinomialDoublyStochastic { n };
        let a = m.unpack(y);
        let (r, c) = m.row_col_sums(&a);
        for i in 0..n {
            assert!((r[i] - 1.0).abs() < tol, "row{i} {}", r[i]);
            assert!((c[i] - 1.0).abs() < tol, "col{i} {}", c[i]);
        }
    }

    #[test]
    fn retract_stays_doubly_stochastic() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = barycenter(2);
        let v = array![0.1, -0.1, -0.1, 0.1];
        let y = m.retract(&x, &v);
        assert_doubly_stochastic(&y, 2, 1e-12);
        let _ = vecops::sum(y.view());
    }

    #[test]
    fn retract_from_off_center_stays_on_set() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = array![0.7, 0.3, 0.3, 0.7];
        let v = array![0.05, -0.05, -0.05, 0.05];
        let y = m.retract(&x, &v);
        assert_doubly_stochastic(&y, 2, 1e-10);
    }

    #[test]
    fn project_is_tangent() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = barycenter(2);
        let v = array![1.0, 2.0, 3.0, 4.0];
        let t = m.project(&x, &v);
        let r0 = t[0] + t[1];
        let r1 = t[2] + t[3];
        let c0 = t[0] + t[2];
        let c1 = t[1] + t[3];
        assert!(r0.abs() < 1e-12, "row0 {r0}");
        assert!(r1.abs() < 1e-12, "row1 {r1}");
        assert!(c0.abs() < 1e-12, "col0 {c0}");
        assert!(c1.abs() < 1e-12, "col1 {c1}");
    }

    #[test]
    fn project_is_fisher_orthogonal() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = array![0.7, 0.3, 0.3, 0.7];
        let v = array![1.0, 0.0, 0.0, 0.0];
        let t = m.project(&x, &v);
        let r0 = t[0] + t[1];
        let r1 = t[2] + t[3];
        let c0 = t[0] + t[2];
        let c1 = t[1] + t[3];
        assert!(r0.abs() < 1e-10, "row0 {r0}");
        assert!(r1.abs() < 1e-10, "row1 {r1}");
        assert!(c0.abs() < 1e-10, "col0 {c0}");
        assert!(c1.abs() < 1e-10, "col1 {c1}");
        // Residual ./ X is of the form alpha_i + beta_j.
        let resid = [
            (v[0] - t[0]) / x[0],
            (v[1] - t[1]) / x[1],
            (v[2] - t[2]) / x[2],
            (v[3] - t[3]) / x[3],
        ];
        let cycle = resid[0] + resid[3] - resid[1] - resid[2];
        assert!(cycle.abs() < 1e-10, "not Fisher dual residual {resid:?}");
        // Euclidean (X-free) projection is a different vector here.
        let euc00 = v[0] - 0.5 - 0.5 + 0.25;
        assert!((t[0] - euc00).abs() > 1e-6, "collapsed to Euclidean {t:?}");
    }

    #[test]
    fn egrad2rgrad_scales_by_x_then_projects() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = array![0.7, 0.3, 0.3, 0.7];
        let g = array![1.0, -1.0, 0.5, 0.0];
        let r = m.egrad2rgrad(&x, &g);
        let mu = array![x[0] * g[0], x[1] * g[1], x[2] * g[2], x[3] * g[3]];
        let p = m.project(&x, &mu);
        for i in 0..4 {
            assert!((r[i] - p[i]).abs() < 1e-12, "r={r:?} p={p:?}");
        }
    }

    #[test]
    fn pack_unpack_round_trip() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let mat = array![[0.7, 0.3], [0.3, 0.7]];
        let p = MultinomialDoublyStochastic::pack(&mat);
        let u = m.unpack(&p);
        assert_eq!(p, array![0.7, 0.3, 0.3, 0.7]);
        assert!((u[(0, 0)] - 0.7).abs() < 1e-15);
        assert!((u[(0, 1)] - 0.3).abs() < 1e-15);
        assert!((u[(1, 0)] - 0.3).abs() < 1e-15);
        assert!((u[(1, 1)] - 0.7).abs() < 1e-15);
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = MultinomialDoublyStochastic { n: 2 };
        let x = barycenter(2);
        let y = array![0.7, 0.3, 0.3, 0.7];
        let v = array![0.1, -0.1, -0.1, 0.1];
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        assert!((&t - &p).mapv(f64::abs).sum() < 1e-14);
    }

    #[test]
    fn required_dim_is_n_squared() {
        let m = MultinomialDoublyStochastic { n: 3 };
        assert_eq!(m.required_dim(9), Ok(()));
        assert_eq!(m.required_dim(3), Err(9));
        assert_eq!(m.required_dim(4), Err(9));
    }

    #[test]
    fn kind_is_not_sphere_or_reserved() {
        use crate::manifold::ManifoldKind;
        assert_ne!(
            ManifoldKind::MultinomialDoublyStochastic { n: 2 },
            ManifoldKind::Sphere
        );
        assert_ne!(
            ManifoldKind::MultinomialDoublyStochastic { n: 2 },
            ManifoldKind::Stiefel
        );
        assert_eq!(
            ManifoldKind::MultinomialDoublyStochastic { n: 3 }.as_str(),
            "multinomialdoublystochastic"
        );
    }
}
