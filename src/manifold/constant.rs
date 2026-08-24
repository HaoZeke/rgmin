//! A 0-dimensional manifold: one fixed point of length \(n\).
//! manopt `constantfactory`.
//!
//! The tangent space is \(\{0\}\). Projection of any ambient vector is
//! the zero vector. Retraction ignores the increment and returns the
//! point. Transport is the zero vector. Useful as a product factor.
//! This is not a constraint solver: the point is whatever `x` is
//! handed in, and the increment is discarded.

use ndarray::Array1;

use super::Manifold;

/// One fixed point of ambient length \(n \ge 1\).
#[derive(Clone, Copy, Debug, Default)]
pub struct Constant;

impl Manifold for Constant {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 1 { Ok(()) } else { Err(1) }
    }

    fn project(&self, _x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Array1::zeros(v.len())
    }

    fn retract(&self, x: &Array1<f64>, _v: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }

    fn transport(
        &self,
        _x_from: &Array1<f64>,
        _x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::zeros(v.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn project_is_zero() {
        let x = array![1.0, -2.0, 3.0];
        let v = array![0.5, 0.25, -1.0];
        let t = Constant.project(&x, &v);
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|a| *a == 0.0));
    }

    #[test]
    fn retract_ignores_the_increment() {
        let x = array![1.5, -0.25, 4.0];
        let v = array![10.0, 20.0, -30.0];
        let y = Constant.retract(&x, &v);
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
    }

    #[test]
    fn transport_is_zero() {
        let x = array![0.0, 1.0];
        let y = array![2.0, 3.0];
        let v = array![0.4, -0.7];
        let t = Constant.transport(&x, &y, &v);
        assert_eq!(t.len(), 2);
        assert!(t.iter().all(|a| *a == 0.0));
    }

    #[test]
    fn required_dim_accepts_any_positive_length() {
        assert!(Constant.required_dim(1).is_ok());
        assert!(Constant.required_dim(7).is_ok());
        assert_eq!(Constant.required_dim(0), Err(1));
    }

    #[test]
    fn kind_is_not_euclidean() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::Constant, ManifoldKind::Euclidean);
        assert_eq!(ManifoldKind::Constant.as_str(), "constant");
        let x = array![1.0, 2.0];
        let v = array![0.3, -0.1];
        let p = ManifoldKind::Constant.project(&x, &v);
        let e = ManifoldKind::Euclidean.project(&x, &v);
        assert!((&e - &v).mapv(f64::abs).sum() < 1e-15);
        assert!(p.iter().all(|a| *a == 0.0));
    }

    #[test]
    fn retract_does_not_solve_a_constraint() {
        let x = array![2.0, 0.0];
        let v = array![-1.0, 1.0];
        let y = Constant.retract(&x, &v);
        assert!((y[0] - 2.0).abs() < 1e-15);
        assert!(y[1].abs() < 1e-15);
        let nrm = (y[0] * y[0] + y[1] * y[1]).sqrt();
        assert!((nrm - 1.0).abs() > 0.5, "must not project onto a sphere");
    }
}
