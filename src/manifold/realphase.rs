//! Product of 0-spheres \(\{\pm 1\}^n\). manopt `realphasefactory`.
//!
//! Each coordinate lives on \(S^0 = \{z \in \mathbb{R} : |z| = 1\}\).
//! The product is 0-dimensional: the tangent space at each \(\pm 1\)
//! is \(\{0\}\), so projection is the zero vector. Retraction stays
//! on \(\{\pm 1\}\) (the step is ignored; there is no discrete hop).
//! Transport is the zero vector at the arrival point. This is the
//! real sign product, not the Fourier-phase submanifold of
//! `complexcirclefactory` and not a 3N cluster.

use ndarray::Array1;

use super::Manifold;

/// Product of `n` copies of \(S^0\). Packed length `n`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RealPhase;

impl RealPhase {
    /// Identity pack of a sign vector.
    pub fn pack(x: Array1<f64>) -> Array1<f64> {
        x
    }

    /// Inverse of [`Self::pack`].
    pub fn unpack(x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }
}

/// Snap a real to \(\{+1, -1\}\). Zero and NaN map to \(+1\).
fn sign_unit(x: f64) -> f64 {
    if x.is_sign_negative() { -1.0 } else { 1.0 }
}

/// `true` when every entry has absolute value 1.
pub fn is_realphase(x: &Array1<f64>) -> bool {
    !x.is_empty() && x.iter().all(|&xi| (xi.abs() - 1.0).abs() <= 1e-14)
}

impl Manifold for RealPhase {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n >= 1 { Ok(()) } else { Err(1) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != v.len() {
            return Array1::zeros(v.len());
        }
        Array1::zeros(v.len())
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let _ = v;
        Array1::from_iter(x.iter().map(|&xi| sign_unit(xi)))
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
    fn retract_stays_on_signs() {
        let x = array![1.0, -1.0, 1.0, -1.0];
        let v = array![10.0, -4.0, 0.5, 100.0];
        let y = RealPhase.retract(&x, &v);
        assert_eq!(y.len(), 4);
        assert!(is_realphase(&y), "left {{+/-1}}^n {y:?}");
        assert!((y[0] - 1.0).abs() < 1e-15, "{y:?}");
        assert!((y[1] + 1.0).abs() < 1e-15, "{y:?}");
        assert!((y[2] - 1.0).abs() < 1e-15, "{y:?}");
        assert!((y[3] + 1.0).abs() < 1e-15, "{y:?}");
    }

    #[test]
    fn retract_does_not_hop() {
        let x = array![1.0, -1.0];
        let v = array![-2.0, 2.0];
        let y = RealPhase.retract(&x, &v);
        assert!((y[0] - 1.0).abs() < 1e-15, "hopped {y:?}");
        assert!((y[1] + 1.0).abs() < 1e-15, "hopped {y:?}");
    }

    #[test]
    fn project_is_zero() {
        let x = array![1.0, -1.0, 1.0];
        let v = array![0.3, -1.2, 4.0];
        let t = RealPhase.project(&x, &v);
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|&ti| ti.abs() < 1e-15), "{t:?}");
    }

    #[test]
    fn transport_is_zero_at_arrival() {
        let x = array![1.0, -1.0];
        let y = array![-1.0, 1.0];
        let v = array![0.5, -0.25];
        let t = RealPhase.transport(&x, &y, &v);
        assert!(t.iter().all(|&ti| ti.abs() < 1e-15), "{t:?}");
    }

    #[test]
    fn required_dim_rejects_empty() {
        assert_eq!(RealPhase.required_dim(0), Err(1));
        assert!(RealPhase.required_dim(1).is_ok());
        assert!(RealPhase.required_dim(114).is_ok());
    }

    #[test]
    fn kind_token_is_realphase() {
        assert_eq!(
            crate::manifold::ManifoldKind::RealPhase.as_str(),
            "realphase"
        );
        assert_ne!(
            crate::manifold::ManifoldKind::RealPhase,
            crate::manifold::ManifoldKind::Sphere
        );
    }
}
