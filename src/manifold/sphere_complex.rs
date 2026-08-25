//! Complex unit sphere \(\{z \in \mathbb{C}^n : z^* z = 1\}\).
//!
//! manopt `spherecomplexfactory` (default \(m = 1\)): unit Frobenius
//! norm in \(\mathbb{C}^n\). Packed interleaved `(re, im)`, length
//! `2 n`. The real inner product \(\mathrm{Re}(x^* u)\) is the
//! Euclidean product on the packed vector, so the embedding is the
//! real sphere in \(\mathbb{R}^{2n}\). Projection is
//! `v - Re(x^* v) x`. Retraction is `(x+v)/||x+v||`. Transport is
//! projection at arrival. This token is not [`super::Sphere`] (real
//! \(S^{n-1}\)) and not [`super::ComplexCircle`] (product of \(S^1\)).
//! Reserved tokens 7-10 stay unused.
//!
//! Reductions go through [`crate::vecops`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Unit sphere in \(\mathbb{C}^n\), packed as `2 n` reals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereComplex {
    /// Complex dimension. Packed length is `2 n`.
    pub n: usize,
}

impl Default for SphereComplex {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl SphereComplex {
    /// \(\mathbb{C}^n\) unit sphere. Illegal `n == 0` fails
    /// [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Packed length `2 n`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(2)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.packed_len() == Some(len)
    }

    /// Interleaved pack of real and imaginary parts (equal length).
    pub fn pack(re: ArrayView1<f64>, im: ArrayView1<f64>) -> Array1<f64> {
        pack(re, im)
    }

    /// Split a packed vector into `(re, im)`. `None` if the length is odd.
    pub fn unpack(x: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
        unpack(x)
    }
}

/// Interleaved pack of real and imaginary parts (equal length).
pub fn pack(re: ArrayView1<f64>, im: ArrayView1<f64>) -> Array1<f64> {
    let n = re.len().min(im.len());
    let mut out = Array1::zeros(2 * n);
    for k in 0..n {
        out[2 * k] = re[k];
        out[2 * k + 1] = im[k];
    }
    out
}

/// Split a packed vector into `(re, im)`. `None` if the length is odd.
pub fn unpack(x: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
    if x.len() % 2 != 0 {
        return None;
    }
    let n = x.len() / 2;
    let mut re = Array1::zeros(n);
    let mut im = Array1::zeros(n);
    for k in 0..n {
        re[k] = x[2 * k];
        im[k] = x[2 * k + 1];
    }
    Some((re, im))
}

/// manopt `M.inner = real(d1(:)'*d2(:))`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = pi`.
pub fn typical_dist() -> f64 {
    std::f64::consts::PI
}

/// `true` when `x` is a legal packed point on the complex sphere.
pub fn is_sphere_complex(x: &Array1<f64>) -> bool {
    x.len() >= 2 && x.len() % 2 == 0 && (vecops::nrm2(x.view()) - 1.0).abs() < 1e-12
}

impl Manifold for SphereComplex {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= 1 && n == want => Ok(()),
            Some(want) => Err(want),
            None => Err(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return v.clone();
        }
        let s = vecops::dot(x.view(), v.view());
        let mut out = v.clone();
        vecops::axpy(-s, x.view(), &mut out);
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return x.clone();
        }
        let mut y = Vector::from_host(x.clone());
        vecops::vaxpy(1.0, &Vector::from_host(v.clone()), &mut y);
        let n = vecops::vnrm2(&y);
        if n <= 1e-16 {
            let n0 = vecops::nrm2(x.view());
            if n0 <= 1e-16 {
                return x.clone();
            }
            return x / n0;
        }
        let mut yh = y.into_host();
        yh.mapv_inplace(|yi| yi / n);
        yh
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_complex_sphere() {
        let m = SphereComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 0.0];
        let v = array![0.0, 0.1, 0.2, -0.1];
        let y = m.retract(&x, &v);
        assert!(is_sphere_complex(&y), "left the complex sphere {y:?}");
        let n2 = vecops::nrm2(y.view());
        assert!((n2 - 1.0).abs() < 1e-14, "norm {n2}");
    }

    #[test]
    fn project_is_hermitian_tangent() {
        let m = SphereComplex { n: 2 };
        let x = array![0.6, 0.8, 0.0, 0.0];
        let v = array![1.0, 2.0, 3.0, 4.0];
        let t = m.project(&x, &v);
        let ip = inner(&x, &t);
        assert!(ip.abs() < 1e-14, "Re(x^* t) = {ip}");
    }

    #[test]
    fn required_dim_is_twice_complex_n() {
        let m = SphereComplex { n: 3 };
        assert_eq!(m.required_dim(6), Ok(()));
        assert_eq!(m.required_dim(3), Err(6));
        assert!(SphereComplex::new(1).required_dim(2).is_ok());
        assert!(SphereComplex::new(0).required_dim(0).is_err());
    }

    #[test]
    fn not_the_real_sphere_token() {
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::ComplexCircle { n: 2 }
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::EuclideanComplex { n: 2 }
        );
    }

    #[test]
    fn pack_unpack_round_trips() {
        let re = array![1.0, 0.0, -1.0];
        let im = array![0.0, 1.0, 0.5];
        let x = SphereComplex::pack(re.view(), im.view());
        assert_eq!(x, array![1.0, 0.0, 0.0, 1.0, -1.0, 0.5]);
        let (r2, i2) = SphereComplex::unpack(&x).unwrap();
        assert!((r2 - re).mapv(f64::abs).sum() < 1e-15);
        assert!((i2 - im).mapv(f64::abs).sum() < 1e-15);
        assert!(SphereComplex::unpack(&array![1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn inner_is_real_dot_and_typical_dist_is_pi() {
        let u = array![1.0, 0.0, 0.0, 0.0];
        let v = array![0.0, 1.0, 0.0, 0.0];
        assert!(inner(&u, &v).abs() < 1e-15);
        assert!((typical_dist() - std::f64::consts::PI).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = SphereComplex { n: 2 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(4));
        assert!(m.required_dim(4).is_ok());
    }

    #[test]
    fn transport_of_a_tangent_stays_tangent() {
        let m = SphereComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 0.0];
        let v = m.project(&x, &array![0.2, 0.3, -0.1, 0.4]);
        let y = m.retract(&x, &v);
        let t = m.transport(&x, &y, &v);
        let ip = inner(&y, &t);
        assert!(ip.abs() < 1e-14, "Re(y^* t) = {ip}");
    }
}
