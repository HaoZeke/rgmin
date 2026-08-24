//! Unit sphere \(S^{n-1}\). manopt `spherefactory(n)` (`m = 1`).
//!
//! Projection \(v - (x\cdot v)x\). Retraction \((x+v)/\|x+v\|\).
//! Transport is projection at the arrival point. The inner product
//! is the Frobenius / Euclidean dot (`d1(:).'*d2(:)`). Typical
//! distance is \(\pi\). Unit-Frobenius \(n \times m\) matrices
//! (`spherefactory(n, m)` with `m > 1`) are a different factory.

use ndarray::Array1;

use crate::vecops;

use super::Manifold;

/// Unit sphere in the ambient Euclidean metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sphere;

/// Frobenius inner product. manopt `M.inner = d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = @() pi`. Diameter of the unit sphere.
pub fn typical_dist() -> f64 {
    std::f64::consts::PI
}

impl Manifold for Sphere {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut t = v.clone();
        let s = vecops::dot(x.view(), v.view());
        vecops::axpy(-s, x.view(), &mut t);
        t
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let mut y = x.clone();
        vecops::axpy(1.0, v.view(), &mut y);
        let n = vecops::nrm2(y.view());
        if n <= 1e-16 {
            let n0 = vecops::nrm2(x.view());
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
    fn project_is_tangent() {
        let x = array![1.0, 0.0, 0.0];
        let v = array![2.0, 3.0, 4.0];
        let t = Sphere.project(&x, &v);
        assert!(inner(&x, &t).abs() < 1e-15, "x·t = {}", inner(&x, &t));
        assert!((t[1] - 3.0).abs() < 1e-15);
        assert!((t[2] - 4.0).abs() < 1e-15);
        assert!(t[0].abs() < 1e-15);
        let r = Sphere.egrad2rgrad(&x, &v);
        for i in 0..3 {
            assert!((r[i] - t[i]).abs() < 1e-15, "egrad2rgrad {r:?} != {t:?}");
        }
        let tt = Sphere.project(&x, &t);
        for i in 0..3 {
            assert!((tt[i] - t[i]).abs() < 1e-15, "P^2 != P {tt:?} != {t:?}");
        }
    }

    #[test]
    fn retract_stays_on_the_sphere() {
        let x = array![0.0, 1.0, 0.0];
        let v = array![0.1, 0.0, -0.2];
        let y = Sphere.retract(&x, &v);
        assert!(
            (vecops::nrm2(y.view()) - 1.0).abs() < 1e-14,
            "left S^{n-1} {y:?}"
        );
        let z = Sphere.retract(&x, &Array1::zeros(3));
        assert!((vecops::nrm2(z.view()) - 1.0).abs() < 1e-15);
        for i in 0..3 {
            assert!((z[i] - x[i]).abs() < 1e-15, "zero step {z:?} != {x:?}");
        }
    }

    #[test]
    fn transport_stays_tangent_at_arrival() {
        let x = array![1.0, 0.0, 0.0];
        let v = Sphere.project(&x, &array![0.0, 0.3, -0.4]);
        let y = Sphere.retract(&x, &v);
        let w = array![0.2, -0.1, 0.5];
        let t = Sphere.transport(&x, &y, &w);
        assert!(
            inner(&y, &t).abs() < 1e-14,
            "arrival x·T(v) = {} at {y:?} t={t:?}",
            inner(&y, &t)
        );
        let p = Sphere.project(&y, &w);
        for i in 0..3 {
            assert!(
                (t[i] - p[i]).abs() < 1e-15,
                "transp != proj(y) {t:?} != {p:?}"
            );
        }
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![1.0, 2.0, 3.0];
        let v = array![4.0, -1.0, 0.5];
        assert!((inner(&u, &v) - 3.5).abs() < 1e-15);
        assert!((typical_dist() - std::f64::consts::PI).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn not_oblique_and_not_a_quotient() {
        assert_ne!(
            crate::manifold::ManifoldKind::Sphere,
            crate::manifold::ManifoldKind::oblique(3, 1)
        );
        assert_ne!(
            crate::manifold::ManifoldKind::Sphere,
            crate::manifold::ManifoldKind::Euclidean
        );
        assert_ne!(
            crate::manifold::ManifoldKind::Sphere,
            crate::manifold::ManifoldKind::RigidQuotient
        );
        assert_eq!(crate::manifold::ManifoldKind::Sphere.as_str(), "sphere");
    }
}
