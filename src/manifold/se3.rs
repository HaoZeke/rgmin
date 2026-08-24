//! SE(3): row-major SO(3) (9) then translation (3). Length 12.
//!
//! manopt `specialeuclideanfactory(3)` at `k = 1`. A point is the
//! product of a 3-by-3 rotation (row-major) and a translation in
//! \(\mathbb{R}^3\). This is the product geometry
//! \(\mathrm{SO}(3)\times\mathbb{R}^3\) (manopt `productmanifold` of
//! `rotationsfactory` and `euclideanfactory`), not the Lie /
//! semidirect geometry of \(\mathrm{SE}(3)\).
//!
//! Tangent vectors use the ambient embedding: the rotation block is
//! \(R\Omega\) with \(\Omega^\top = -\Omega\), and the translation is
//! Euclidean. Projection is \(R\,\mathrm{skew}(R^\top H)\) on the
//! rotation and the identity on \(t\). Retraction is `qr_unique` of
//! \(R + V\) (manopt `retr_qr`) plus \(t + u\). Transport is
//! projection at the arrival point. The inner product is the
//! Frobenius / Euclidean dot (`d1(:).'*d2(:)`). Typical distance is
//! \(\sqrt{3(\pi^2+1)}\), the product of `pi*sqrt(3)` and `sqrt(3)`.
//!
//! Dimension is exactly 12. A 3N cluster is
//! [`super::RigidQuotient`], not a 12-vector prefix. This factory
//! does not become SE(n) for `n != 3`.

use ndarray::Array1;

use crate::vecops;

use super::{so3::So3, Manifold};

/// Rigid motions. Rotation block uses [`So3`]; translation is Euclidean.
#[derive(Clone, Copy, Debug, Default)]
pub struct Se3;

/// Row-major rotation then translation as a length-12 token.
pub fn pack(r: [[f64; 3]; 3], t: [f64; 3]) -> Array1<f64> {
    Array1::from_shape_vec(12, {
        let mut v = Vec::with_capacity(12);
        for i in 0..3 {
            for j in 0..3 {
                v.push(r[i][j]);
            }
        }
        v.extend_from_slice(&t);
        v
    })
    .unwrap()
}

/// Split a packed point or tangent into `(R, t)`.
pub fn unpack(x: &Array1<f64>) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut r = [[0.0; 3]; 3];
    let mut t = [0.0; 3];
    if x.len() >= 9 {
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = x[3 * i + j];
            }
        }
    }
    if x.len() >= 12 {
        t = [x[9], x[10], x[11]];
    }
    (r, t)
}

/// Product Frobenius / Euclidean inner product. manopt ambient dot.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt product of `typicaldist`: \(\sqrt{\pi^2\cdot 3 + 3}\).
pub fn typical_dist() -> f64 {
    (3.0 * (std::f64::consts::PI * std::f64::consts::PI + 1.0)).sqrt()
}

/// `true` when the packed point is in \(\mathrm{SE}(3)\).
pub fn is_se3(x: &Array1<f64>) -> bool {
    if x.len() != 12 {
        return false;
    }
    let (r, t) = unpack(x);
    t.iter().all(|a| a.is_finite()) && is_so3_block(r)
}

fn is_so3_block(r: [[f64; 3]; 3]) -> bool {
    let rtr = mul(transpose(r), r);
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            if (rtr[i][j] - want).abs() > 1e-8 {
                return false;
            }
        }
    }
    det(r) > 0.0
}

fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn det(q: [[f64; 3]; 3]) -> f64 {
    q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1])
        - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
        + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0])
}

impl Manifold for Se3 {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == 12 {
            Ok(())
        } else {
            Err(12)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 12 || v.len() != 12 {
            return v.clone();
        }
        let xr = x.slice(ndarray::s![0..9]).to_owned();
        let vr = v.slice(ndarray::s![0..9]).to_owned();
        let pr = So3.project(&xr, &vr);
        let mut out = v.clone();
        for i in 0..9 {
            out[i] = pr[i];
        }
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 12 || v.len() != 12 {
            return x + v;
        }
        let xr = x.slice(ndarray::s![0..9]).to_owned();
        let vr = v.slice(ndarray::s![0..9]).to_owned();
        let yr = So3.retract(&xr, &vr);
        let mut y = x.clone();
        vecops::axpy(1.0, v.view(), &mut y);
        for i in 0..9 {
            y[i] = yr[i];
        }
        y
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn eye_t(tx: f64, ty: f64, tz: f64) -> Array1<f64> {
        pack(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [tx, ty, tz],
        )
    }

    #[test]
    fn translation_is_euclidean() {
        let x = eye_t(1.0, 2.0, 3.0);
        let v = pack([[0.0; 3]; 3], [0.4, -0.1, 0.2]);
        let y = Se3.retract(&x, &v);
        assert!((y[9] - 1.4).abs() < 1e-15);
        assert!((y[10] - 1.9).abs() < 1e-15);
        assert!((y[11] - 3.2).abs() < 1e-15);
        let (r, t) = unpack(&y);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((r[i][j] - want).abs() < 1e-14);
            }
        }
        assert!((t[0] - 1.4).abs() < 1e-15);
        assert!(is_se3(&y));
    }

    #[test]
    fn rotation_block_stays_so3() {
        let x = eye_t(0.0, 0.0, 0.0);
        let mut v = Array1::zeros(12);
        v[1] = -0.2;
        v[3] = 0.2;
        v[9] = 1.0;
        let y = Se3.retract(&x, &v);
        let (r, t) = unpack(&y);
        let rtr = mul(transpose(r), r);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - want).abs() < 1e-12, "{rtr:?}");
            }
        }
        assert!((det(r) - 1.0).abs() < 1e-12, "det {d}", d = det(r));
        assert!((t[0] - 1.0).abs() < 1e-15);
        assert!(t[1].abs() < 1e-15);
        assert!(t[2].abs() < 1e-15);
        assert!(is_se3(&y));
    }

    #[test]
    fn project_rotation_is_skew_translation_is_identity() {
        let x = eye_t(0.2, -0.1, 0.4);
        let v = array![0.2, 0.1, -0.3, 0.4, -0.1, 0.5, 0.0, 0.2, 0.3, 0.7, -0.2, 0.1];
        let z = Se3.project(&x, &v);
        assert_eq!(z.len(), 12);
        let (r, _) = unpack(&x);
        let (body, tz) = unpack(&z);
        let omega = mul(transpose(r), body);
        for i in 0..3 {
            for j in 0..3 {
                let s = omega[i][j] + omega[j][i];
                assert!(s.abs() < 1e-12, "R^T Z not skew [{i},{j}] = {s}");
            }
        }
        assert!((tz[0] - 0.7).abs() < 1e-15);
        assert!((tz[1] + 0.2).abs() < 1e-15);
        assert!((tz[2] - 0.1).abs() < 1e-15);
        let rgrad = Se3.egrad2rgrad(&x, &v);
        for i in 0..12 {
            assert!((rgrad[i] - z[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn retract_stays_in_se3() {
        let x = eye_t(0.5, -0.2, 0.1);
        let mut raw = Array1::zeros(12);
        raw[1] = -0.2;
        raw[3] = 0.2;
        raw[9] = 0.3;
        let v = Se3.project(&x, &raw);
        let y = Se3.retract(&x, &v);
        assert!(is_se3(&y), "left SE(3) {y:?}");
        assert_eq!(y.len(), 12);
        assert!((y[9] - 0.8).abs() < 1e-15);
        let (r, _) = unpack(&y);
        assert!((det(r) - 1.0).abs() < 1e-12, "det {d}", d = det(r));
        let z = Se3.retract(&x, &Array1::zeros(12));
        assert!((&z - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_se3(&z));
    }

    #[test]
    fn a_3n_cluster_is_refused() {
        assert!(Se3.required_dim(12).is_ok());
        assert!(Se3.required_dim(9).is_err());
        assert!(Se3.required_dim(6).is_err());
        assert!(Se3.required_dim(3).is_err());
        assert!(Se3.required_dim(114).is_err());
        assert_eq!(Se3.required_dim(114).unwrap_err(), 12);
        assert!(!is_se3(&Array1::from_elem(114, 0.1)));
        assert!(!is_se3(&Array1::from_elem(6, 0.1)));
        assert_ne!(
            crate::manifold::ManifoldKind::Se3,
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::Se3,
            crate::manifold::ManifoldKind::RigidQuotient
        );
        assert_ne!(
            crate::manifold::ManifoldKind::Se3,
            crate::manifold::ManifoldKind::So3
        );
        assert_ne!(
            crate::manifold::ManifoldKind::Se3,
            crate::manifold::ManifoldKind::MwRigid
        );
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let x = eye_t(0.0, 0.0, 0.0);
        let mut raw = Array1::zeros(12);
        raw[1] = -0.15;
        raw[3] = 0.15;
        raw[9] = 0.3;
        let v = Se3.project(&x, &raw);
        let y = Se3.retract(&x, &v);
        let t = Se3.transport(&x, &y, &v);
        let p = Se3.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(is_se3(&y));
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let v = array![0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert!((inner(&u, &v) - 3.0).abs() < 1e-15);
        let want = (3.0 * (std::f64::consts::PI * std::f64::consts::PI + 1.0)).sqrt();
        assert!((typical_dist() - want).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_is_row_major_then_t() {
        let r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let t = [3.0, 4.0, 5.0];
        let x = pack(r, t);
        assert_eq!(x.len(), 12);
        assert_eq!(
            x.as_slice().unwrap(),
            &[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 5.0]
        );
        let (back_r, back_t) = unpack(&x);
        assert_eq!(back_r, r);
        assert_eq!(back_t, t);
        assert!(is_se3(&x));
    }

    #[test]
    fn wrong_dim_is_identity_and_keeps_length() {
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = Se3.retract(&x, &v);
        assert_eq!(y.len(), 114);
        for i in 0..114 {
            assert!((y[i] - (x[i] + v[i])).abs() < 1e-15);
        }
        assert_eq!(Se3.project(&x, &v).len(), 114);
        assert!(Se3.required_dim(114).is_err());
        assert!(Se3.required_dim(12).is_ok());
    }
}
