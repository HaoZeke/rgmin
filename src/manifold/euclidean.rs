//! Ambient Euclidean space. manopt `euclideanfactory` on a rank-1 vector.
//!
//! Projection and transport are the identity. Retraction is `x + v`.
//! The inner product is the Frobenius / Euclidean dot
//! (`d1(:).'*d2(:)`). Typical distance is `sqrt(n)` for a packed
//! length-`n` vector (`sqrt(prod(size))` at `euclideanfactory(n)`).
//! Matrix or tensor packing is a different factory.

use ndarray::Array1;

use crate::vecops;

use super::Manifold;

/// Unconstrained Euclidean geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

/// Frobenius inner product. manopt `M.inner = d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = sqrt(prod(dimensions_vec))` for `R^n`.
pub fn typical_dist(n: usize) -> f64 {
    (n as f64).sqrt()
}

impl Manifold for Euclidean {
    fn project(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        v.to_owned()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut y = x.clone();
        vecops::axpy(1.0, v.view(), &mut y);
        y
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        v.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn project_is_identity() {
        let x = array![1.0, -2.0, 0.5];
        let v = array![0.3, 4.0, -1.0];
        let t = Euclidean.project(&x, &v);
        assert_eq!(t.len(), v.len());
        for i in 0..3 {
            assert!((t[i] - v[i]).abs() < 1e-15, "project {t:?} != {v:?}");
        }
        let r = Euclidean.egrad2rgrad(&x, &v);
        for i in 0..3 {
            assert!((r[i] - v[i]).abs() < 1e-15, "egrad2rgrad {r:?} != {v:?}");
        }
    }

    #[test]
    fn retract_is_translation() {
        let x = array![1.0, 2.0];
        let v = array![0.5, -1.0];
        let y = Euclidean.retract(&x, &v);
        assert!((y[0] - 1.5).abs() < 1e-15);
        assert!((y[1] - 1.0).abs() < 1e-15);
        assert_eq!(y.len(), x.len());
    }

    #[test]
    fn transport_is_identity() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![-4.0, 0.0, 8.0];
        let v = array![0.25, -0.5, 1.0];
        let t = Euclidean.transport(&x, &y, &v);
        for i in 0..3 {
            assert!((t[i] - v[i]).abs() < 1e-15, "transport {t:?} != {v:?}");
        }
    }

    #[test]
    fn retract_stays_on_the_euclidean_set() {
        let x = array![-1.2, 3.4, 0.0];
        let v = array![0.1, -0.2, 5.0];
        let y = Euclidean.retract(&x, &v);
        assert_eq!(y.len(), 3);
        assert!(y.iter().all(|a| a.is_finite()), "left R^n {y:?}");
        assert!(Euclidean.required_dim(y.len()).is_ok());
        assert!(Euclidean.required_dim(0).is_ok());
        assert!(Euclidean.required_dim(114).is_ok());
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![1.0, 2.0, 3.0];
        let v = array![4.0, -1.0, 0.5];
        assert!((inner(&u, &v) - 3.5).abs() < 1e-15);
        assert!((typical_dist(4) - 2.0).abs() < 1e-15);
        assert!((typical_dist(1) - 1.0).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn not_the_sphere_and_not_a_quotient() {
        let x = array![2.0, 0.0, 0.0];
        let v = array![0.0, 0.3, 0.0];
        let y = Euclidean.retract(&x, &v);
        let n2: f64 = y.iter().map(|a| a * a).sum();
        assert!((n2 - 1.0).abs() > 1.0, "must not be a unit sphere {y:?}");
        assert_ne!(
            crate::manifold::ManifoldKind::Euclidean,
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::Euclidean,
            crate::manifold::ManifoldKind::RigidQuotient
        );
        assert_eq!(
            crate::manifold::ManifoldKind::default(),
            crate::manifold::ManifoldKind::Euclidean
        );
    }
}
