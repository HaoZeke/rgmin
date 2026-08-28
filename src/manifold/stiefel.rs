//! Stiefel \(\mathrm{St}(n,1)\): a single orthonormal column is the sphere.
//!
//! `p > 1` is not a length-\(n\) token. A 3N cluster is
//! [`super::RigidQuotient`], not \(\mathrm{St}(3N, 1)\).

use ndarray::Array1;

use super::{Manifold, sphere::Sphere};

/// Stiefel with \(p=1\), packed as a length-\(n\) unit vector.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stiefel;

impl Manifold for Stiefel {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Sphere.project(x, v)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Sphere.retract(x, v)
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Sphere.transport(x_from, x_to, v)
    }
}
