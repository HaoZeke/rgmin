//! Doubly-centered real matrices. manopt `centeredmatrixfactory`.
//!
//! A point is an `m x n` real matrix packed row-major as length
//! `m*n`. The default geometry removes both row means and column
//! means (the two-way residual). Projection is the closed-form
//! centering \(Y_{ij} = X_{ij} - r_i - c_j + \mu\). Retraction is
//! `X + U` then re-center. Transport is projection at the arrival
//! point. A 3N cluster is not this packing unless `(m, n)` says so.

use ndarray::{Array1, Array2};

use super::Manifold;

/// Doubly-centered `m x n` matrices. Packed row-major, length `m*n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Centered {
    /// Number of rows.
    pub m: usize,
    /// Number of columns.
    pub n: usize,
}

impl Centered {
    /// Doubly-centered `m x n` matrices.
    pub fn new(m: usize, n: usize) -> Self {
        Self { m, n }
    }

    /// Packed length `m*n`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.m.checked_mul(self.n)
    }

    fn fits(self, len: usize) -> bool {
        self.m >= 1 && self.n >= 1 && self.packed_len() == Some(len)
    }

    /// Row-major flatten of an m-by-n matrix.
    pub fn pack(mat: &Array2<f64>) -> Array1<f64> {
        let (m, n) = mat.dim();
        let mut out = Array1::zeros(m * n);
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = mat[[i, j]];
            }
        }
        out
    }

    /// Inverse of [`Self::pack`] for this `(m, n)`.
    pub fn unpack(&self, x: &Array1<f64>) -> Array2<f64> {
        let mut mat = Array2::zeros((self.m, self.n));
        if !self.fits(x.len()) {
            return mat;
        }
        for i in 0..self.m {
            for j in 0..self.n {
                mat[[i, j]] = x[i * self.n + j];
            }
        }
        mat
    }
}

impl Default for Centered {
    fn default() -> Self {
        Self { m: 2, n: 2 }
    }
}

/// Subtract row means and column means in one shot.
///
/// \(Y_{ij} = X_{ij} - r_i - c_j + \mu\) with \(r_i\) the mean of
/// row \(i\), \(c_j\) the mean of column \(j\), and \(\mu\) the
/// grand mean. Both row sums and column sums of \(Y\) are zero.
fn center_both(m: usize, n: usize, a: &[f64]) -> Vec<f64> {
    let mut row_mean = vec![0.0; m];
    let mut col_mean = vec![0.0; n];
    let mut grand = 0.0;
    for i in 0..m {
        for j in 0..n {
            let v = a[i * n + j];
            row_mean[i] += v;
            col_mean[j] += v;
            grand += v;
        }
    }
    let nf = n as f64;
    let mf = m as f64;
    for r in row_mean.iter_mut() {
        *r /= nf;
    }
    for c in col_mean.iter_mut() {
        *c /= mf;
    }
    grand /= mf * nf;
    let mut y = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] = a[i * n + j] - row_mean[i] - col_mean[j] + grand;
        }
    }
    y
}

/// Largest absolute row-mean or column-mean of a packed matrix.
pub fn max_mean_abs(m: usize, n: usize, a: &[f64]) -> f64 {
    if m == 0 || n == 0 || a.len() != m * n {
        return 0.0;
    }
    let mut worst = 0.0_f64;
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j];
        }
        worst = worst.max((s / n as f64).abs());
    }
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..m {
            s += a[i * n + j];
        }
        worst = worst.max((s / m as f64).abs());
    }
    worst
}

impl Manifold for Centered {
    fn required_dim(&self, dim: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.m >= 1 && self.n >= 1 && dim == want => Ok(()),
            Some(want) => Err(want),
            None => Err(dim),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || v.len() != x.len() {
            return v.clone();
        }
        let flat: Vec<f64> = v.iter().copied().collect();
        Array1::from(center_both(self.m, self.n, &flat))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || v.len() != x.len() {
            return x + v;
        }
        let flat: Vec<f64> = x.iter().zip(v.iter()).map(|(xi, vi)| *xi + *vi).collect();
        Array1::from(center_both(self.m, self.n, &flat))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn two_by_three() -> Centered {
        Centered::new(2, 3)
    }

    #[test]
    fn project_has_zero_row_and_column_means() {
        let m = two_by_three();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = array![1.0, 0.0, -1.0, 2.0, 3.0, 4.0];
        let t = m.project(&x, &v);
        assert_eq!(t.len(), 6);
        let flat: Vec<f64> = t.iter().copied().collect();
        let worst = max_mean_abs(2, 3, &flat);
        assert!(worst < 1e-14, "means {worst} t={t:?}");
    }

    #[test]
    fn retract_stays_doubly_centered() {
        let m = two_by_three();
        let x = m.project(&Array1::zeros(6), &array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = array![0.1, -0.2, 0.3, -0.4, 0.5, -0.3];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 6);
        let flat: Vec<f64> = y.iter().copied().collect();
        let worst = max_mean_abs(2, 3, &flat);
        assert!(worst < 1e-14, "means {worst} y={y:?}");
    }

    #[test]
    fn centering_is_closed_form() {
        // 2x2: [[1, 3], [2, 6]]
        let x = array![1.0, 3.0, 2.0, 6.0];
        let m = Centered::new(2, 2);
        let y = m.project(&x, &x);
        // row means [2, 4], col means [1.5, 4.5], mu = 3
        // Y00 = 1 - 2 - 1.5 + 3 = 0.5
        // Y01 = 3 - 2 - 4.5 + 3 = -0.5
        // Y10 = 2 - 4 - 1.5 + 3 = -0.5
        // Y11 = 6 - 4 - 4.5 + 3 = 0.5
        assert!((y[0] - 0.5).abs() < 1e-15, "{y:?}");
        assert!((y[1] + 0.5).abs() < 1e-15, "{y:?}");
        assert!((y[2] + 0.5).abs() < 1e-15, "{y:?}");
        assert!((y[3] - 0.5).abs() < 1e-15, "{y:?}");
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = two_by_three();
        let x = Array1::zeros(6);
        let y = m.retract(&x, &array![1.0, -1.0, 0.5, -0.5, 0.25, -0.25]);
        let v = array![2.0, 1.0, 0.0, -1.0, -2.0, 3.0];
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn pack_unpack_is_row_major() {
        let m = two_by_three();
        let mat = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = Centered::pack(&mat);
        assert_eq!(x, array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let back = m.unpack(&x);
        assert_eq!(back, mat);
    }

    #[test]
    fn required_dim_rejects_a_3n_cluster() {
        let m = Centered::new(2, 3);
        assert!(m.required_dim(6).is_ok());
        assert_eq!(m.required_dim(114), Err(6));
        assert_eq!(m.required_dim(9), Err(6));
        assert!(Centered::new(1, 1).required_dim(1).is_ok());
        assert!(Centered::new(0, 4).required_dim(0).is_err());
    }

    #[test]
    fn wrong_dim_keeps_length() {
        let m = two_by_three();
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        for i in 0..114 {
            assert!((y[i] - (x[i] + v[i])).abs() < 1e-15);
        }
        assert_eq!(m.project(&x, &v).len(), 114);
    }

    #[test]
    fn kind_token_is_centered() {
        assert_eq!(
            crate::manifold::ManifoldKind::centered(2, 3).as_str(),
            "centered"
        );
        assert_ne!(
            crate::manifold::ManifoldKind::centered(2, 3),
            crate::manifold::ManifoldKind::Oblique { n: 2, m: 3 }
        );
    }
}
