//! Complex Euclidean space \(\mathbb{C}^n\) as \(\mathbb{R}^{2n}\).
//! manopt `euclideancomplexfactory`.
//!
//! Packed interleaved `(re, im)` pairs, length `2 n`. The geometry is
//! Euclidean: project is the identity, retract is `x + v`, transport
//! is the identity. Distinct from [`super::ComplexCircle`] (unit
//! modulus per pair) and from [`super::Sphere`] (one constraint on
//! the whole vector). Packed length must be even and at least 2.

use ndarray::{Array1, ArrayView1};

use super::{Euclidean, Manifold};

/// \(\mathbb{C}^n\) identified with \(\mathbb{R}^{2n}\) via interleaved pairs.
#[derive(Clone, Copy, Debug, Default)]
pub struct EuclideanComplex;

impl EuclideanComplex {
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
}

impl Manifold for EuclideanComplex {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 2 && n % 2 == 0 {
            Ok(())
        } else if n < 2 {
            Err(2)
        } else {
            Err(n + 1)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Euclidean.project(x, v)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Euclidean.retract(x, v)
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Euclidean.transport(x_from, x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_is_translation() {
        let x = array![1.0, 2.0, -0.5, 0.25];
        let v = array![0.5, -1.0, 0.1, 0.0];
        let y = EuclideanComplex.retract(&x, &v);
        assert!((y[0] - 1.5).abs() < 1e-15);
        assert!((y[1] - 1.0).abs() < 1e-15);
        assert!((y[2] + 0.4).abs() < 1e-15);
        assert!((y[3] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn project_and_transport_are_identity() {
        let x = array![1.0, 0.0, 0.0, 1.0];
        let y = array![0.5, 0.5, -0.2, 0.8];
        let v = array![0.3, -0.1, 0.2, 0.4];
        let t = EuclideanComplex.project(&x, &v);
        assert!((&t - &v).mapv(f64::abs).sum() < 1e-15);
        let w = EuclideanComplex.transport(&x, &y, &v);
        assert!((&w - &v).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn retract_leaves_the_unit_circle() {
        let x = array![1.0, 0.0];
        let v = array![0.3, 0.4];
        let y = EuclideanComplex.retract(&x, &v);
        let nrm = (y[0] * y[0] + y[1] * y[1]).sqrt();
        assert!((nrm - 1.0).abs() > 0.2, "must not stay on S^1 {y:?}");
    }

    #[test]
    fn retract_does_not_normalize_like_the_sphere() {
        let x = array![1.0, 0.0, 0.0, 0.0];
        let v = array![0.0, 1.0, 0.0, 0.0];
        let y = EuclideanComplex.retract(&x, &v);
        let nrm = y.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!((nrm - 2.0_f64.sqrt()).abs() < 1e-14);
        assert!((nrm - 1.0).abs() > 0.3, "must not be S^3 {y:?}");
    }

    #[test]
    fn pack_unpack_round_trips() {
        let re = array![1.0, 0.0, -1.0];
        let im = array![0.0, 1.0, 0.5];
        let x = EuclideanComplex::pack(re.view(), im.view());
        assert_eq!(x, array![1.0, 0.0, 0.0, 1.0, -1.0, 0.5]);
        let (r2, i2) = EuclideanComplex::unpack(&x).unwrap();
        assert!((r2 - re).mapv(f64::abs).sum() < 1e-15);
        assert!((i2 - im).mapv(f64::abs).sum() < 1e-15);
        assert!(EuclideanComplex::unpack(&array![1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn required_dim_is_even_and_at_least_two() {
        assert!(EuclideanComplex.required_dim(2).is_ok());
        assert!(EuclideanComplex.required_dim(4).is_ok());
        assert_eq!(EuclideanComplex.required_dim(0), Err(2));
        assert_eq!(EuclideanComplex.required_dim(1), Err(2));
        assert_eq!(EuclideanComplex.required_dim(3), Err(4));
        assert_eq!(EuclideanComplex.required_dim(5), Err(6));
    }

    #[test]
    fn kind_is_not_complex_circle_or_sphere() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::EuclideanComplex, ManifoldKind::Sphere);
        assert_ne!(
            ManifoldKind::EuclideanComplex,
            ManifoldKind::ComplexCircle { n: 1 }
        );
        assert_eq!(ManifoldKind::EuclideanComplex.as_str(), "euclidean_complex");
    }

    #[test]
    fn zero_step_is_the_point() {
        let x = array![0.3, -0.7, 1.2, 0.0];
        let y = EuclideanComplex.retract(&x, &Array1::zeros(4));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
    }
}
