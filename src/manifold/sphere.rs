//! Unit sphere \(S^{n-1}\). manopt_cpp `Sphere`.
//!
//! Projection \(v - (x\cdot v)x\). Retraction \((x+v)/\|x+v\|\).
//! Transport is projection at the arrival point.

use ndarray::Array1;

use super::Manifold;

/// Unit sphere in the ambient Euclidean metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sphere;

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

fn nrm(a: &Array1<f64>) -> f64 {
    dot(a, a).sqrt()
}

impl Manifold for Sphere {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let s = dot(x, v);
        Array1::from_iter(x.iter().zip(v.iter()).map(|(xi, vi)| vi - s * xi))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let y = x + v;
        let n = nrm(&y);
        if n <= 1e-16 {
            let n0 = nrm(x);
            if n0 <= 1e-16 {
                return x.clone();
            }
            return x / n0;
        }
        y / n
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
    fn project_is_orthogonal_to_x() {
        let x = array![1.0, 0.0, 0.0];
        let v = array![2.0, 3.0, 4.0];
        let t = Sphere.project(&x, &v);
        assert!(dot(&x, &t).abs() < 1e-15);
        assert!((t[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn retract_stays_on_the_sphere() {
        let x = array![0.0, 1.0, 0.0];
        let v = array![0.1, 0.0, -0.2];
        let y = Sphere.retract(&x, &v);
        assert!((nrm(&y) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn rgrad_is_the_tangent_projection() {
        let x = array![1.0, 0.0, 0.0];
        let egrad = array![2.0, 3.0, 4.0];
        let r = Sphere.egrad2rgrad(&x, &egrad);
        let t = Sphere.project(&x, &egrad);
        assert!((&r - &t).mapv(f64::abs).sum() < 1e-15);
        assert!(dot(&x, &r).abs() < 1e-15);
    }
}
