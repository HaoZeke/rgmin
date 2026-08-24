//! Strictly positive orthant. manopt `positivefactory`.
//!
//! A point is \(x \in \mathbb{R}^n\) with \(x_i > 0\). The geometry is
//! the open set with the componentwise log / Hessian-of-barrier
//! metric. Projection is the identity (the orthant is open).
//! Retraction is \(x \odot \exp(v \oslash x)\), which stays strictly
//! positive. Transport is the identity. A 3N cluster is a valid
//! packing of this factory: positivity is a pointwise constraint,
//! not a molecular quotient.

use ndarray::Array1;

use super::Manifold;

/// Strictly positive vectors in \(\mathbb{R}^n\).
#[derive(Clone, Copy, Debug, Default)]
pub struct Positive;

impl Positive {
    /// Identity pack of a positive vector.
    pub fn pack(x: Array1<f64>) -> Array1<f64> {
        x
    }

    /// Inverse of [`Self::pack`].
    pub fn unpack(x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }
}

/// `true` when every entry is strictly positive.
pub fn is_positive(x: &Array1<f64>) -> bool {
    !x.is_empty() && x.iter().all(|&xi| xi > 0.0 && xi.is_finite())
}

impl Manifold for Positive {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 1 { Ok(()) } else { Err(1) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() {
            return v.clone();
        }
        v.clone()
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() {
            return x + v;
        }
        Array1::from_iter(x.iter().zip(v.iter()).map(|(xi, vi)| {
            let den = xi.max(f64::EPSILON);
            let yi = xi.max(f64::EPSILON) * (*vi / den).exp();
            if yi.is_finite() {
                yi.max(f64::EPSILON)
            } else {
                f64::EPSILON
            }
        }))
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
    fn retract_stays_positive() {
        let x = array![0.2, 1.5, 3.0];
        let v = array![0.4, -2.0, 1.0];
        let y = Positive.retract(&x, &v);
        assert_eq!(y.len(), 3);
        assert!(is_positive(&y), "left the positive orthant {y:?}");
        let want0 = 0.2_f64 * (0.4_f64 / 0.2_f64).exp();
        let want1 = 1.5_f64 * (-2.0_f64 / 1.5_f64).exp();
        let want2 = 3.0_f64 * (1.0_f64 / 3.0_f64).exp();
        assert!((y[0] - want0).abs() < 1e-14, "{y:?}");
        assert!((y[1] - want1).abs() < 1e-14, "{y:?}");
        assert!((y[2] - want2).abs() < 1e-14, "{y:?}");
    }

    #[test]
    fn large_negative_step_stays_positive() {
        let x = array![1e-3, 2.0];
        let v = array![-10.0, -50.0];
        let y = Positive.retract(&x, &v);
        assert!(is_positive(&y), "left the positive orthant {y:?}");
        assert!(y[0] < x[0], "expected a shrink {y:?}");
        assert!(y[1] < x[1], "expected a shrink {y:?}");
    }

    #[test]
    fn project_is_identity() {
        let x = array![1.0, 2.0];
        let v = array![-3.0, 4.5];
        let t = Positive.project(&x, &v);
        assert!((t[0] + 3.0).abs() < 1e-15);
        assert!((t[1] - 4.5).abs() < 1e-15);
    }

    #[test]
    fn transport_of_a_tangent_is_itself() {
        let x = array![1.0, 2.0];
        let y = array![1.5, 0.5];
        let v = array![0.25, -0.1];
        let t = Positive.transport(&x, &y, &v);
        assert!((t[0] - 0.25).abs() < 1e-15);
        assert!((t[1] + 0.1).abs() < 1e-15);
    }

    #[test]
    fn required_dim_rejects_empty() {
        assert_eq!(Positive.required_dim(0), Err(1));
        assert!(Positive.required_dim(1).is_ok());
        assert!(Positive.required_dim(114).is_ok());
    }

    #[test]
    fn length_mismatch_does_not_shrink() {
        let x = array![1.0, 2.0];
        let v = array![0.1];
        let y = Positive.retract(&x, &v);
        assert_eq!(y.len(), 2);
        assert_eq!(Positive.project(&x, &v).len(), 1);
    }

    #[test]
    fn kind_token_is_positive() {
        assert_eq!(crate::manifold::ManifoldKind::Positive.as_str(), "positive");
        assert_ne!(
            crate::manifold::ManifoldKind::Positive,
            crate::manifold::ManifoldKind::Euclidean
        );
    }
}
