//! Symmetric doubly-stochastic matrices with the Fisher metric.
//!
//! manopt `multinomialsymmetricfactory`. The point is an n-by-n
//! symmetric matrix, packed row-major, with positive entries and
//! unit row (hence column) sums. The tangent space is the symmetric
//! matrices with zero row sums. Retraction is `X ⊙ exp(V ⊘ X)`,
//! symmetrized, then Sinkhorn. Transport is projection at arrival.
//!
//! Distinct from [`super::Multinomial`], from
//! [`super::MultinomialDoublyStochastic`], and from the reserved
//! tokens 7-10.

use ndarray::{Array1, Array2};

use super::Manifold;

/// Symmetric Birkhoff polytope, Fisher information metric.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialSymmetric {
    /// Side length. Packed length is `n * n`.
    pub n: usize,
}

impl MultinomialSymmetric {
    fn mat(&self, packed: &Array1<f64>) -> Array2<f64> {
        let n = self.n;
        let mut a = Array2::<f64>::zeros((n, n));
        if packed.len() == n * n {
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = packed[i * n + j];
                }
            }
        }
        a
    }

    fn pack_mat(&self, a: &Array2<f64>) -> Array1<f64> {
        let n = self.n;
        let mut p = Array1::zeros(n * n);
        for i in 0..n {
            for j in 0..n {
                p[i * n + j] = a[(i, j)];
            }
        }
        p
    }

    fn symmetrize(&self, mut a: Array2<f64>) -> Array2<f64> {
        let n = self.n;
        for i in 0..n {
            for j in i..n {
                let s = 0.5 * (a[(i, j)] + a[(j, i)]);
                a[(i, j)] = s;
                a[(j, i)] = s;
            }
        }
        a
    }

    fn sinkhorn(&self, mut a: Array2<f64>) -> Array2<f64> {
        let n = self.n;
        for _ in 0..64 {
            for i in 0..n {
                let mut rs = 0.0;
                for j in 0..n {
                    a[(i, j)] = a[(i, j)].max(f64::EPSILON);
                    rs += a[(i, j)];
                }
                if rs > 0.0 && rs.is_finite() {
                    for j in 0..n {
                        a[(i, j)] /= rs;
                    }
                }
            }
            a = self.symmetrize(a);
        }
        // One last column-consistent Sinkhorn pass after symmetrize.
        for j in 0..n {
            let mut cs = 0.0;
            for i in 0..n {
                cs += a[(i, j)];
            }
            if cs > 0.0 && cs.is_finite() {
                for i in 0..n {
                    a[(i, j)] /= cs;
                }
            }
        }
        self.symmetrize(a)
    }
}

impl Manifold for MultinomialSymmetric {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.n >= 2 && n == self.n * self.n {
            Ok(())
        } else {
            Err(self.n.max(2) * self.n.max(2))
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let n = self.n;
        if x.len() != n * n || v.len() != n * n {
            return v.clone();
        }
        let vm = self.symmetrize(self.mat(v));
        let mut r = Array1::<f64>::zeros(n);
        let mut total = 0.0;
        for i in 0..n {
            for j in 0..n {
                r[i] += vm[(i, j)];
                total += vm[(i, j)];
            }
        }
        let shift = total / (n * n) as f64;
        let mut out = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] = vm[(i, j)] - r[i] / n as f64 - r[j] / n as f64 + shift;
            }
        }
        self.pack_mat(&out)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let n = self.n;
        if x.len() != n * n || v.len() != n * n {
            return x + v;
        }
        let xm = self.mat(x);
        let vm = self.mat(v);
        let mut y = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let xij = xm[(i, j)].max(f64::EPSILON);
                y[(i, j)] = xij * (vm[(i, j)] / xij).exp();
            }
        }
        self.pack_mat(&self.sinkhorn(self.symmetrize(y)))
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

    #[test]
    fn retract_stays_symmetric_doubly_stochastic() {
        let m = MultinomialSymmetric { n: 2 };
        let x = barycenter(2);
        let v = array![0.0, 0.1, 0.1, -0.2];
        let y = m.retract(&x, &v);
        assert!(y.iter().all(|&yi| yi > 0.0), "left the interior {y:?}");
        assert!((y[1] - y[2]).abs() < 1e-12, "not symmetric {y:?}");
        let r0 = y[0] + y[1];
        let r1 = y[2] + y[3];
        let c0 = y[0] + y[2];
        assert!((r0 - 1.0).abs() < 1e-10, "row0 {r0}");
        assert!((r1 - 1.0).abs() < 1e-10, "row1 {r1}");
        assert!((c0 - 1.0).abs() < 1e-10, "col0 {c0}");
    }

    #[test]
    fn project_is_symmetric_and_tangent() {
        let m = MultinomialSymmetric { n: 2 };
        let x = barycenter(2);
        let v = array![1.0, 2.0, 3.0, 4.0];
        let t = m.project(&x, &v);
        assert!((t[1] - t[2]).abs() < 1e-14, "not symmetric {t:?}");
        let r0 = t[0] + t[1];
        let r1 = t[2] + t[3];
        assert!(r0.abs() < 1e-14, "row0 {r0}");
        assert!(r1.abs() < 1e-14, "row1 {r1}");
    }

    #[test]
    fn required_dim_is_n_squared() {
        let m = MultinomialSymmetric { n: 3 };
        assert_eq!(m.required_dim(9), Ok(()));
        assert_eq!(m.required_dim(6), Err(9));
    }
}
