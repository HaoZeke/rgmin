//! Positive orthant \(\mathbb{R}_{++}^n\). manopt `positivefactory`.
//!
//! A point is a strictly positive n-vector: manopt `positivefactory(n)`
//! with the second dimension defaulting to 1. An m-by-k matrix is
//! packed column-major as `X(:)`, length `n = m k`. The tangent space
//! is all of \(\mathbb{R}^n\) (open subset). The metric is the
//! scale-invariant product \(\langle u, v\rangle_x = (u \oslash x)^\top
//! (v \oslash x)\). Projection is the identity. Retraction is the
//! exponential \(x \odot \exp(v \oslash x)\). Transport is the identity
//! (manopt default). Euclidean-to-Riemannian gradient is
//! \(x \odot \mathrm{egrad} \odot x\). This token is not the sphere,
//! not SPD, and not a 3N cluster packing. Reserved tokens 7-10 stay
//! unused.
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, Array2, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Strictly positive n-vector. manopt `positivefactory(n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Positive {
    /// Packed length. manopt `m*n` with default second dim 1.
    pub n: usize,
}

impl Default for Positive {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl Positive {
    /// Positive orthant of length `n`. Illegal `n == 0` fails
    /// [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.n == len
    }

    /// Column-major flatten of an m-by-k matrix (manopt `X(:)`).
    pub fn pack(mat: &Array2<f64>) -> Array1<f64> {
        pack(mat)
    }

    /// Inverse of [`Self::pack`] for this packed length as `m`-by-`k`.
    pub fn unpack(&self, x: &Array1<f64>, m: usize, k: usize) -> Option<Array2<f64>> {
        unpack(x, m, k)
    }
}

/// Column-major flatten of an m-by-k matrix (manopt `X(:)`).
pub fn pack(mat: &Array2<f64>) -> Array1<f64> {
    let (m, k) = mat.dim();
    let mut out = Array1::zeros(m * k);
    for j in 0..k {
        for i in 0..m {
            out[i + j * m] = mat[[i, j]];
        }
    }
    out
}

/// Inverse of [`pack`]. `None` if `m*k` overflows or mismatches `x`.
pub fn unpack(x: &Array1<f64>, m: usize, k: usize) -> Option<Array2<f64>> {
    let want = m.checked_mul(k)?;
    if m < 1 || k < 1 || x.len() != want {
        return None;
    }
    let mut mat = Array2::zeros((m, k));
    for j in 0..k {
        for i in 0..m {
            mat[[i, j]] = x[i + j * m];
        }
    }
    Some(mat)
}

/// Identity pack of a positive vector (manopt `M.vec` at `n = 1`).
pub fn pack_vec(x: ArrayView1<f64>) -> Array1<f64> {
    x.to_owned()
}

/// Identity unpack of a positive vector (manopt `M.mat` at `n = 1`).
pub fn unpack_vec(x: &Array1<f64>) -> Array1<f64> {
    x.clone()
}

/// Scale-invariant inner product. manopt `(eta./X)'*(zeta./X)`.
pub fn inner(x: &Array1<f64>, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    let n = x.len().min(u.len()).min(v.len());
    let mut uhat = Vector::zeros_cpu(n);
    let mut vhat = Vector::zeros_cpu(n);
    {
        let uh = uhat.host_mut();
        let vh = vhat.host_mut();
        for i in 0..n {
            let xi = x[i];
            if xi.abs() > 0.0 {
                uh[i] = u[i] / xi;
                vh[i] = v[i] / xi;
            }
        }
    }
    vecops::vdot(&uhat, &vhat)
}

/// manopt `M.typicaldist = sqrt(m*n)`.
pub fn typical_dist(n: usize) -> f64 {
    (n as f64).sqrt()
}

/// `true` when every entry is strictly positive and the vector is nonempty.
pub fn is_positive(x: &Array1<f64>) -> bool {
    !x.is_empty() && x.iter().all(|xi| xi.is_finite() && *xi > 0.0)
}

impl Manifold for Positive {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.n >= 1 && n == self.n {
            Ok(())
        } else {
            Err(self.n)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return v.clone();
        }
        v.clone()
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != egrad.len() {
            return egrad.clone();
        }
        let mut out = Vector::from_host(egrad.clone());
        for (oi, xi) in out.host_mut().iter_mut().zip(x.iter()) {
            *oi *= *xi * *xi;
        }
        out.into_host()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() || !is_positive(x) {
            return x.clone();
        }
        let mut y = Vector::from_host(x.clone());
        for (yi, vi) in y.host_mut().iter_mut().zip(v.iter()) {
            let step = (*vi / *yi).exp();
            *yi *= step;
            if !yi.is_finite() || *yi <= 0.0 {
                *yi = f64::MIN_POSITIVE;
            }
        }
        y.into_host()
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        v.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_set() {
        let m = Positive { n: 3 };
        let x = array![1.0, 2.0, 0.5];
        let v = array![0.1, -0.5, 0.2];
        let y = m.retract(&x, &v);
        assert!(is_positive(&y), "left the positive orthant {y:?}");
        assert_eq!(y.len(), 3);
        let want0 = 1.0 * (0.1_f64 / 1.0).exp();
        let want1 = 2.0 * ((-0.5_f64) / 2.0).exp();
        let want2 = 0.5 * (0.2_f64 / 0.5).exp();
        assert!((y[0] - want0).abs() < 1e-14, "y0 {y:?}");
        assert!((y[1] - want1).abs() < 1e-14, "y1 {y:?}");
        assert!((y[2] - want2).abs() < 1e-14, "y2 {y:?}");
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
    }

    #[test]
    fn project_is_the_ambient_tangent() {
        let m = Positive { n: 2 };
        let x = array![1.0, 2.0];
        let v = array![0.3, -0.4];
        let t = m.project(&x, &v);
        assert!((&t - &v).mapv(f64::abs).sum() < 1e-15);
        assert!((inner(&x, &v, &v) - (0.3 * 0.3 + 0.2 * 0.2)).abs() < 1e-15);
        assert!((typical_dist(2) - 2.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn egrad_scales_by_x_squared() {
        let m = Positive { n: 2 };
        let x = array![1.0, 2.0];
        let g = array![1.0, 1.0];
        let r = m.egrad2rgrad(&x, &g);
        assert!((r[0] - 1.0).abs() < 1e-15);
        assert!((r[1] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn transport_is_the_identity() {
        let m = Positive { n: 2 };
        let x = array![1.0, 2.0];
        let y = array![1.5, 0.5];
        let v = array![0.2, -0.1];
        let t = m.transport(&x, &y, &v);
        assert!((&t - &v).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn pack_unpack_round_trips() {
        let mat = array![[1.0, 3.0], [2.0, 4.0]];
        let x = Positive::pack(&mat);
        assert_eq!(x, array![1.0, 2.0, 3.0, 4.0]);
        let back = Positive { n: 4 }.unpack(&x, 2, 2).unwrap();
        assert!((&back - &mat).mapv(f64::abs).sum() < 1e-15);
        assert!(Positive { n: 4 }.unpack(&x, 3, 2).is_none());
        let v = array![1.25, 0.5, 3.0];
        assert_eq!(pack_vec(v.view()), v);
        assert_eq!(unpack_vec(&v), v);
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = Positive { n: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert!(
            (&y - &x).mapv(f64::abs).sum() < 1e-15,
            "must not apply the exp map to a cluster"
        );
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(2));
        assert!(m.required_dim(2).is_ok());
        assert!(Positive::new(1).required_dim(1).is_ok());
        assert!(Positive::new(0).required_dim(0).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = Positive { n: 3 };
        let x = array![1.0, 0.5, 2.0];
        let y = m.retract(&x, &Array1::zeros(3));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_positive(&y));
    }

    #[test]
    fn kind_is_not_sphere_or_stiefel() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::Positive { n: 3 }, ManifoldKind::Sphere);
        assert_ne!(ManifoldKind::Positive { n: 1 }, ManifoldKind::Stiefel);
        assert_ne!(ManifoldKind::Positive { n: 3 }, ManifoldKind::Euclidean);
        assert_ne!(ManifoldKind::Positive { n: 4 }, ManifoldKind::Spd);
    }
}
