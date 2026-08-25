//! Singleton manifold of one fixed point. manopt `constantfactory(A)`.
//!
//! The manifold is the 0-dimensional set `{A}`. A point is packed as
//! the ambient vector `A` itself, length `n >= 1`. The only tangent
//! vector is the zero array of that length: projection and transport
//! vanish through [`crate::vecops`]. Retraction discards the increment
//! and returns the point (`retr(x, v) = x`). The inner product and
//! typical distance are 0.
//!
//! This is a helper geometry (product factors that stay pinned). It is
//! not the sphere, not Euclidean, and not a 3N cluster packing. Isolated
//! molecules use [`super::RigidQuotient`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, Vector};

use super::Manifold;

/// Singleton `{A}` of packed length `n >= 1`. manopt `constantfactory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constant {
    /// Ambient length of the fixed point.
    pub n: usize,
}

impl Default for Constant {
    fn default() -> Self {
        Self { n: 1 }
    }
}

impl Constant {
    /// A singleton of length `n`. Illegal `n == 0` fails [`Manifold::required_dim`].
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 1 && self.n == len
    }

    /// Pack the singleton. The packed form is the point itself.
    pub fn pack(a: ArrayView1<f64>) -> Array1<f64> {
        pack(a)
    }

    /// Unpack the singleton. Identity of the packed vector.
    pub fn unpack(x: &Array1<f64>) -> Array1<f64> {
        unpack(x)
    }
}

/// Pack the singleton. The packed form is the point itself.
pub fn pack(a: ArrayView1<f64>) -> Array1<f64> {
    a.to_owned()
}

/// Unpack the singleton. Identity of the packed vector.
pub fn unpack(x: &Array1<f64>) -> Array1<f64> {
    x.clone()
}

/// manopt `M.inner = 0` on a 0-dimensional tangent space.
pub fn inner(_u: &Array1<f64>, _v: &Array1<f64>) -> f64 {
    0.0
}

/// manopt `M.typicaldist = 0`.
pub fn typical_dist() -> f64 {
    0.0
}

/// `true` when `x` is a legal singleton packing (`length = n >= 1`).
pub fn is_constant(x: &Array1<f64>) -> bool {
    !x.is_empty()
}

impl Manifold for Constant {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.n >= 1 && n == self.n {
            Ok(())
        } else if self.n >= 1 {
            Err(self.n)
        } else {
            Err(1)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return Vector::zeros_cpu(v.len()).into_host();
        }
        Vector::zeros_cpu(v.len()).into_host()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(x.len()) || x.len() != v.len() {
            return x.clone();
        }
        Vector::from_host(x.clone()).into_host()
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        Vector::zeros_cpu(v.len()).into_host()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_set() {
        let m = Constant { n: 3 };
        let x = array![1.5, -0.25, 4.0];
        let v = array![10.0, 20.0, -30.0];
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 3);
        assert!(is_constant(&y), "left the singleton {y:?}");
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        let fro = vecops::nrm2(y.view());
        assert!((fro - 1.0).abs() > 0.5, "must not be the sphere {y:?}");
    }

    #[test]
    fn project_is_zero() {
        let m = Constant { n: 3 };
        let x = array![1.0, -2.0, 3.0];
        let v = array![0.5, 0.25, -1.0];
        let t = m.project(&x, &v);
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|a| *a == 0.0));
        assert!(vecops::nrm2(t.view()) == 0.0);
    }

    #[test]
    fn transport_of_a_tangent_is_zero() {
        let m = Constant { n: 2 };
        let x = array![0.0, 1.0];
        let y = array![2.0, 3.0];
        let v = array![0.4, -0.7];
        let t = m.transport(&x, &y, &v);
        assert_eq!(t.len(), 2);
        assert!(t.iter().all(|a| *a == 0.0));
    }

    #[test]
    fn pack_unpack_round_trips() {
        let a = array![1.0, -2.0, 0.5];
        let x = Constant::pack(a.view());
        assert_eq!(x, a);
        let a2 = Constant::unpack(&x);
        assert!((&a2 - &a).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn inner_is_zero_and_typical_dist_is_zero() {
        let u = array![1.0, 2.0, 3.0];
        let v = array![-1.0, 4.0, 0.5];
        assert!(inner(&u, &v).abs() < 1e-15);
        assert!(inner(&u, &u).abs() < 1e-15);
        assert!(typical_dist().abs() < 1e-15);
    }

    #[test]
    fn wrong_dim_rejects_a_3n_cluster() {
        let m = Constant { n: 3 };
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(m.project(&x, &v).len(), 114);
        assert_eq!(m.required_dim(114), Err(3));
        assert!(m.required_dim(3).is_ok());
        assert!(Constant::new(1).required_dim(1).is_ok());
        assert!(Constant::new(0).required_dim(0).is_err());
        assert!(Constant::new(3).required_dim(5).is_err());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = Constant { n: 3 };
        let x = array![1.0, -0.5, 0.25];
        let y = m.retract(&x, &Array1::zeros(3));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn kind_is_not_sphere_or_euclidean() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::Constant { n: 3 }, ManifoldKind::Sphere);
        assert_ne!(ManifoldKind::Constant { n: 3 }, ManifoldKind::Euclidean);
        assert_ne!(ManifoldKind::Constant { n: 3 }, ManifoldKind::Stiefel);
        assert_ne!(
            ManifoldKind::Constant { n: 3 },
            ManifoldKind::EuclideanComplex { n: 3 }
        );
    }
}
