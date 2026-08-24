//! SO(3) as a 9-vector, row-major. manopt `rotationsfactory(3)` at `k = 1`.
//!
//! A point is a 3-by-3 rotation packed row-major, length 9.
//! Tangent vectors are the ambient embedding \(R\Omega\) with
//! \(\Omega^\top = -\Omega\), not the Lie-algebra factor alone
//! (manopt stores the body and maps with `tangent2ambient`).
//! Projection is \(R\,\mathrm{skew}(R^\top H)\). Retraction is
//! `qr_unique` of \(R + V\) (manopt `retr_qr`) with the last column
//! flipped if \(\det < 0\). Transport is projection at the arrival
//! point. The inner product is the Frobenius / Euclidean dot
//! (`d1(:).'*d2(:)`). Typical distance is \(\pi\sqrt{3}\)
//! (`pi*sqrt(n*k)` at `n = 3`, `k = 1`).
//!
//! Dimension is exactly 9. A 3N cluster is
//! [`super::RigidQuotient`], not this embedding. This factory does
//! not become SO(n) for `n != 3`.

use ndarray::Array1;

use crate::vecops;

use super::Manifold;

/// Rotation matrices packed row-major length 9.
#[derive(Clone, Copy, Debug, Default)]
pub struct So3;

/// Row-major 3-by-3 block as a length-9 token.
pub fn unpack(x: &Array1<f64>) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    if x.len() >= 9 {
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = x[3 * i + j];
            }
        }
    }
    r
}

/// Flatten a 3-by-3 matrix into the ambient 9-vector.
pub fn pack(r: [[f64; 3]; 3]) -> Array1<f64> {
    Array1::from_shape_vec(9, {
        let mut v = Vec::with_capacity(9);
        for i in 0..3 {
            for j in 0..3 {
                v.push(r[i][j]);
            }
        }
        v
    })
    .unwrap()
}

/// Frobenius inner product. manopt `M.inner = d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = pi*sqrt(n*k)` with `n = 3`, `k = 1`.
pub fn typical_dist() -> f64 {
    std::f64::consts::PI * 3.0_f64.sqrt()
}

/// `true` when the packed matrix is in \(\mathrm{SO}(3)\).
pub fn is_so3(x: &Array1<f64>) -> bool {
    if x.len() != 9 {
        return false;
    }
    let r = unpack(x);
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

fn skew_of(s: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut k = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            k[i][j] = 0.5 * (s[i][j] - s[j][i]);
        }
    }
    k
}

fn det(q: [[f64; 3]; 3]) -> f64 {
    q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1])
        - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
        + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0])
}

/// QR retraction with \(\det = +1\). manopt `retr_qr` / `qr_unique`.
fn qr_pos(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut q = [[0.0; 3]; 3];
    for j in 0..3 {
        let mut v = [a[0][j], a[1][j], a[2][j]];
        for k in 0..j {
            let mut d = 0.0;
            for i in 0..3 {
                d += q[i][k] * a[i][j];
            }
            for i in 0..3 {
                v[i] -= d * q[i][k];
            }
        }
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 1e-16 {
            for i in 0..3 {
                q[i][j] = v[i] / n;
            }
        }
    }
    if det(q) < 0.0 {
        for i in 0..3 {
            q[i][2] = -q[i][2];
        }
    }
    q
}

impl Manifold for So3 {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == 9 {
            Ok(())
        } else {
            Err(9)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 9 || v.len() != 9 {
            return v.clone();
        }
        let r = unpack(x);
        let h = unpack(v);
        let rt_h = mul(transpose(r), h);
        let omega = skew_of(rt_h);
        pack(mul(r, omega))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 9 || v.len() != 9 {
            return x + v;
        }
        let mut y = x.clone();
        vecops::axpy(1.0, v.view(), &mut y);
        pack(qr_pos(unpack(&y)))
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn identity() -> Array1<f64> {
        array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn project_is_skew_in_the_body() {
        let x = identity();
        let v = array![0.2, 0.1, -0.3, 0.4, -0.1, 0.5, 0.0, 0.2, 0.3];
        let z = So3.project(&x, &v);
        assert_eq!(z.len(), 9);
        let r = unpack(&x);
        let body = mul(transpose(r), unpack(&z));
        for i in 0..3 {
            for j in 0..3 {
                let s = body[i][j] + body[j][i];
                assert!(s.abs() < 1e-12, "R^T Z not skew [{i},{j}] = {s}");
            }
        }
        let rgrad = So3.egrad2rgrad(&x, &v);
        for i in 0..9 {
            assert!((rgrad[i] - z[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn retract_stays_in_so3() {
        let x = identity();
        let v = So3.project(
            &x,
            &array![0.0, -0.2, 0.1, 0.2, 0.0, -0.05, -0.1, 0.05, 0.0],
        );
        let y = So3.retract(&x, &v);
        assert!(is_so3(&y), "left SO(3) {y:?}");
        assert_eq!(y.len(), 9);
        let r = unpack(&y);
        assert!((det(r) - 1.0).abs() < 1e-12, "det {d}", d = det(r));
        let fro2: f64 = y.iter().map(|a| a * a).sum();
        assert!((fro2 - 3.0).abs() < 1e-12, "not a rotation {y:?}");
        let z = So3.retract(&x, &Array1::zeros(9));
        assert!((&z - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_so3(&z));
    }

    #[test]
    fn a_3n_cluster_is_refused() {
        assert!(So3.required_dim(9).is_ok());
        assert!(So3.required_dim(6).is_err());
        assert!(So3.required_dim(3).is_err());
        assert!(So3.required_dim(12).is_err());
        assert!(So3.required_dim(114).is_err());
        assert_eq!(So3.required_dim(114).unwrap_err(), 9);
        assert!(!is_so3(&Array1::from_elem(114, 0.1)));
        assert!(!is_so3(&Array1::from_elem(6, 0.1)));
        assert_ne!(
            crate::manifold::ManifoldKind::So3,
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::So3,
            crate::manifold::ManifoldKind::RigidQuotient
        );
        assert_ne!(
            crate::manifold::ManifoldKind::So3,
            crate::manifold::ManifoldKind::Se3
        );
    }

    #[test]
    fn identity_plus_skew_stays_orthogonal() {
        let x = identity();
        let v = array![0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = So3.retract(&x, &v);
        let r = unpack(&y);
        let rtr = mul(transpose(r), r);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - want).abs() < 1e-12, "{rtr:?}");
            }
        }
        assert!(is_so3(&y));
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let x = identity();
        let v = So3.project(&x, &array![0.0, -0.15, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let y = So3.retract(&x, &v);
        let t = So3.transport(&x, &y, &v);
        let p = So3.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(is_so3(&y));
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = array![0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((inner(&u, &v) - 1.0).abs() < 1e-15);
        let want = std::f64::consts::PI * 3.0_f64.sqrt();
        assert!((typical_dist() - want).abs() < 1e-15);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn pack_unpack_is_row_major() {
        let r = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let x = pack(r);
        assert_eq!(x.len(), 9);
        assert_eq!(
            x.as_slice().unwrap(),
            &[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        let back = unpack(&x);
        assert_eq!(back, r);
        assert!(is_so3(&x));
    }

    #[test]
    fn wrong_dim_does_not_shrink() {
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = So3.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(So3.project(&x, &v).len(), 114);
        assert!(So3.required_dim(114).is_err());
        assert!(So3.required_dim(9).is_ok());
    }
}
