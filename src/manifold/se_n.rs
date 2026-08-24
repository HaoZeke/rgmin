//! Special Euclidean group \(\mathrm{SE}(n)\) for \(n \ge 2\).
//!
//! manopt `specialeuclideanfactory(n)` at `k = 1`. A point is the
//! product of an `n x n` rotation (row-major) and a translation in
//! \(\mathbb{R}^n\), packed as length `n^2 + n` (same convention as
//! [`super::Se3`]). Tangent vectors use the ambient embedding: the
//! rotation block is \(R\Omega\) with \(\Omega^\top = -\Omega\), and
//! the translation is Euclidean.
//!
//! This is the product geometry \(\mathrm{SO}(n) \times \mathbb{R}^n\)
//! (manopt `productmanifold` of `rotationsfactory` and
//! `euclideanfactory`), not the Lie / semidirect geometry of
//! \(\mathrm{SE}(n)\). Projection is \(R\,\mathrm{skew}(R^\top H)\) on
//! the rotation and the identity on the translation. Retraction is
//! `qr_unique` of \(R + V\) (manopt `retr_qr`) with the last column
//! flipped if \(\det < 0\), plus \(t + u\). Transport is projection at
//! the arrival point. Inner products go through [`crate::vecops`].
//!
//! [`super::Se3`] stays the dedicated 12-vector token. This factory
//! does not reuse [`super::Sphere`] and does not pack a 3N cluster
//! as Stiefel. Isolated molecules use [`super::RigidQuotient`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{axpy, dot, nrm2};

use super::Manifold;

/// \(\mathrm{SE}(n)\), \(n \ge 2\). Packed row-major \(R\) then \(t\),
/// length `n^2 + n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeN {
    /// Ambient dimension of the rotation and the translation.
    /// Must be `>= 2`.
    pub n: usize,
}

impl SeN {
    /// \(\mathrm{SE}(n)\) with \(n \ge 2\).
    pub fn new(n: usize) -> Result<Self, usize> {
        if n >= 2 {
            Ok(Self { n })
        } else {
            Err(n)
        }
    }

    /// Packed length `n^2 + n`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n
            .checked_mul(self.n)
            .and_then(|r| r.checked_add(self.n))
    }

    fn rot_len(self) -> Option<usize> {
        self.n.checked_mul(self.n)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 2 && self.packed_len() == Some(len)
    }

    /// Row-major rotation then translation as a length-`n^2+n` token.
    pub fn pack(&self, rot: &[f64], t: &[f64]) -> Option<Array1<f64>> {
        let rl = self.rot_len()?;
        if rot.len() != rl || t.len() != self.n {
            return None;
        }
        let mut v = Vec::with_capacity(rl + self.n);
        v.extend_from_slice(rot);
        v.extend_from_slice(t);
        Some(Array1::from(v))
    }

    /// Split a packed point or tangent into `(R, t)`.
    pub fn unpack(&self, x: &Array1<f64>) -> Option<(Vec<f64>, Vec<f64>)> {
        let rl = self.rot_len()?;
        if !self.fits(x.len()) {
            return None;
        }
        Some((
            x.slice(ndarray::s![0..rl]).to_vec(),
            x.slice(ndarray::s![rl..]).to_vec(),
        ))
    }

    fn col(&self, a: &[f64], j: usize) -> Array1<f64> {
        let mut c = Array1::zeros(self.n);
        for i in 0..self.n {
            c[i] = a[i * self.n + j];
        }
        c
    }

    fn write_col(&self, a: &mut [f64], j: usize, col: &Array1<f64>) {
        for i in 0..self.n {
            a[i * self.n + j] = col[i];
        }
    }

    fn row<'a>(&self, a: &'a [f64], i: usize) -> ArrayView1<'a, f64> {
        ArrayView1::from(&a[i * self.n..(i + 1) * self.n])
    }

    fn mul(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut c = vec![0.0; n * n];
        for j in 0..n {
            let bj = self.col(b, j);
            for i in 0..n {
                c[i * n + j] = dot(self.row(a, i), bj.view());
            }
        }
        c
    }

    fn transpose(&self, a: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut t = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                t[j * n + i] = a[i * n + j];
            }
        }
        t
    }

    fn skew(&self, a: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = 0.5 * (a[i * n + j] - a[j * n + i]);
            }
        }
        s
    }

    /// Gram-Schmidt columns with positive norms, then \(\det = +1\).
    fn qr_pos(&self, a: &mut [f64]) {
        let n = self.n;
        for j in 0..n {
            let mut v = self.col(a, j);
            for k in 0..j {
                let qk = self.col(a, k);
                let r = dot(qk.view(), v.view());
                axpy(-r, qk.view(), &mut v);
            }
            let nrm = nrm2(v.view());
            if nrm > 1e-16 {
                v.mapv_inplace(|t| t / nrm);
            }
            self.write_col(a, j, &v);
        }
        if det(n, a) < 0.0 {
            let last = n - 1;
            for i in 0..n {
                a[i * n + last] = -a[i * n + last];
            }
        }
    }
}

/// Side length if `len` is `n^2 + n` for some \(n \ge 2\).
pub fn side(len: usize) -> Option<usize> {
    if len < 6 {
        return None;
    }
    let disc = 1.0 + 4.0 * (len as f64);
    let n = ((-1.0 + disc.sqrt()) * 0.5).round() as usize;
    if n >= 2 && n.checked_mul(n).and_then(|r| r.checked_add(n)) == Some(len) {
        Some(n)
    } else {
        None
    }
}

/// Split a length-`n^2+n` ambient vector into `(n, R, t)`.
pub fn unpack(x: &Array1<f64>) -> Option<(usize, Vec<f64>, Vec<f64>)> {
    let n = side(x.len())?;
    let rl = n * n;
    Some((
        n,
        x.iter().take(rl).copied().collect(),
        x.iter().skip(rl).copied().collect(),
    ))
}

/// Flatten a row-major rotation and a translation into the ambient vector.
pub fn pack(n: usize, rot: Vec<f64>, t: Vec<f64>) -> Array1<f64> {
    let mut v = rot;
    v.extend(t);
    Array1::from_shape_vec(n * n + n, v).unwrap()
}

/// Product Frobenius / Euclidean inner product. Ambient dot.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    dot(u.view(), v.view())
}

/// manopt product of `typicaldist`: \(\sqrt{\pi^2 n + n}\).
pub fn typical_dist(n: usize) -> f64 {
    let nf = n as f64;
    (nf * (std::f64::consts::PI * std::f64::consts::PI + 1.0)).sqrt()
}

/// `true` when the packed point is in \(\mathrm{SE}(n)\).
pub fn is_se(x: &Array1<f64>) -> bool {
    let Some((n, rot, t)) = unpack(x) else {
        return false;
    };
    let Ok(m) = SeN::new(n) else {
        return false;
    };
    t.iter().all(|a| a.is_finite()) && is_rotation(&m, &rot)
}

fn is_rotation(m: &SeN, a: &[f64]) -> bool {
    let rt = m.transpose(a);
    let rtr = m.mul(&rt, a);
    let n = m.n;
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            if (rtr[i * n + j] - want).abs() > 1e-8 {
                return false;
            }
        }
    }
    det(n, a) > 0.0
}

fn det(n: usize, a: &[f64]) -> f64 {
    let mut m = a.to_vec();
    let mut d = 1.0;
    for k in 0..n {
        let mut piv = k;
        let mut best = m[k * n + k].abs();
        for i in (k + 1)..n {
            let v = m[i * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-16 {
            return 0.0;
        }
        if piv != k {
            for j in 0..n {
                m.swap(k * n + j, piv * n + j);
            }
            d = -d;
        }
        let akk = m[k * n + k];
        d *= akk;
        for i in (k + 1)..n {
            let f = m[i * n + k] / akk;
            for j in k..n {
                m[i * n + j] -= f * m[k * n + j];
            }
        }
    }
    d
}

impl Manifold for SeN {
    fn required_dim(&self, dim: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= 2 && dim == want => Ok(()),
            Some(want) => Err(want),
            None => Err(dim),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(hv)) = (x.as_slice(), v.as_slice()) else {
            return v.clone();
        };
        if !self.fits(xv.len()) || xv.len() != hv.len() {
            return v.clone();
        }
        let rl = self.n * self.n;
        let rt = self.transpose(&xv[..rl]);
        let rth = self.mul(&rt, &hv[..rl]);
        let omega = self.skew(&rth);
        let pr = self.mul(&xv[..rl], &omega);
        let mut out = v.clone();
        for i in 0..rl {
            out[i] = pr[i];
        }
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (x.as_slice(), v.as_slice()) else {
            return x + v;
        };
        if !self.fits(xv.len()) || xv.len() != uv.len() {
            return x + v;
        }
        let rl = self.n * self.n;
        let mut rot = xv[..rl].to_vec();
        for (yi, ui) in rot.iter_mut().zip(uv[..rl].iter()) {
            *yi += *ui;
        }
        self.qr_pos(&mut rot);
        let mut y = x + v;
        for i in 0..rl {
            y[i] = rot[i];
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

    fn identity_t(n: usize, t: &[f64]) -> Array1<f64> {
        let mut rot = vec![0.0; n * n];
        for i in 0..n {
            rot[i * n + i] = 1.0;
        }
        pack(n, rot, t.to_vec())
    }

    fn skew_rot(n: usize, scale: f64) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        if n >= 2 {
            a[1] = -scale;
            a[n] = scale;
        }
        if n >= 4 {
            a[2 * n + 3] = -0.5 * scale;
            a[3 * n + 2] = 0.5 * scale;
        }
        a
    }

    fn ambient_step(n: usize, scale: f64, t: &[f64]) -> Array1<f64> {
        pack(n, skew_rot(n, scale), t.to_vec())
    }

    #[test]
    fn retract_stays_on_se2() {
        let m = SeN::new(2).unwrap();
        let x = identity_t(2, &[0.5, -0.25]);
        let v = m.project(&x, &ambient_step(2, 0.3, &[0.1, 0.2]));
        let y = m.retract(&x, &v);
        assert!(is_se(&y), "left SE(2) {y:?}");
        assert_eq!(y.len(), 6);
        assert!((y[4] - 0.6).abs() < 1e-15);
        assert!((y[5] - (-0.05)).abs() < 1e-15);
        let det2 = y[0] * y[3] - y[1] * y[2];
        assert!((det2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn retract_stays_on_se4() {
        let m = SeN::new(4).unwrap();
        let x = identity_t(4, &[1.0, 0.0, -0.5, 0.25]);
        let v = m.project(&x, &ambient_step(4, 0.2, &[0.0, 0.1, 0.0, -0.1]));
        let y = m.retract(&x, &v);
        assert!(is_se(&y), "left SE(4) {y:?}");
        assert_eq!(y.len(), 20);
        assert!((y[16] - 1.0).abs() < 1e-15);
        assert!((y[17] - 0.1).abs() < 1e-15);
        assert!(det(4, &y.as_slice().unwrap()[..16]) > 0.0);
    }

    #[test]
    fn translation_is_euclidean() {
        let m = SeN::new(2).unwrap();
        let x = identity_t(2, &[1.0, 2.0]);
        let v = pack(2, vec![0.0; 4], vec![0.4, -0.1]);
        let y = m.retract(&x, &v);
        assert!((y[4] - 1.4).abs() < 1e-15);
        assert!((y[5] - 1.9).abs() < 1e-15);
        assert!((y[0] - 1.0).abs() < 1e-14);
        assert!((y[3] - 1.0).abs() < 1e-14);
        assert!(y[1].abs() < 1e-14);
        assert!(y[2].abs() < 1e-14);
    }

    #[test]
    fn project_is_horizontal() {
        let m = SeN::new(4).unwrap();
        let x = identity_t(4, &[0.0; 4]);
        let mut v = Array1::zeros(20);
        for i in 0..20 {
            v[i] = 0.1 * ((i as f64) - 9.0);
        }
        let z = m.project(&x, &v);
        let rt = m.transpose(&x.as_slice().unwrap()[..16]);
        let rth = m.mul(&rt, &z.as_slice().unwrap()[..16]);
        for i in 0..4 {
            for j in 0..4 {
                let s = rth[i * 4 + j] + rth[j * 4 + i];
                assert!(s.abs() < 1e-12, "R^T Z not skew [{i},{j}] = {s}");
            }
        }
        for i in 16..20 {
            assert!((z[i] - v[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = SeN::new(2).unwrap();
        let x = identity_t(2, &[0.0, 0.0]);
        let v = m.project(&x, &ambient_step(2, 0.25, &[0.3, -0.1]));
        let y = m.retract(&x, &v);
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(is_se(&y));
    }

    #[test]
    fn pack_unpack_is_row_major_then_t() {
        let m = SeN::new(2).unwrap();
        let rot = [0.0, -1.0, 1.0, 0.0];
        let t = [3.0, 4.0];
        let x = m.pack(&rot, &t).unwrap();
        assert_eq!(x.len(), 6);
        let (ur, ut) = m.unpack(&x).unwrap();
        assert_eq!(ur, rot);
        assert_eq!(ut, t);
        assert!(m.pack(&[1.0, 0.0, 0.0], &t).is_none());
        let (n, flat_r, flat_t) = unpack(&x).unwrap();
        assert_eq!(n, 2);
        let y = pack(n, flat_r, flat_t);
        for i in 0..6 {
            assert!((x[i] - y[i]).abs() < 1e-15);
        }
        assert_eq!(side(6), Some(2));
        assert_eq!(side(12), Some(3));
        assert_eq!(side(20), Some(4));
        assert!(side(9).is_none());
        assert!(side(114).is_none());
    }

    #[test]
    fn se3_kind_is_not_this_factory() {
        let m = SeN::new(3).unwrap();
        let x = identity_t(3, &[0.1, 0.2, 0.3]);
        let v = m.project(&x, &ambient_step(3, 0.1, &[0.4, 0.0, -0.2]));
        let y = m.retract(&x, &v);
        let z = super::super::Se3.retract(&x, &v);
        for (a, b) in y.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12, "SeN(3) must match Se3 geometry");
        }
        assert_ne!(
            crate::manifold::ManifoldKind::se_n(3),
            crate::manifold::ManifoldKind::Se3
        );
        assert_ne!(
            crate::manifold::ManifoldKind::se_n(2),
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::se_n(2),
            crate::manifold::ManifoldKind::Stiefel
        );
        assert_ne!(
            crate::manifold::ManifoldKind::se_n(2),
            crate::manifold::ManifoldKind::So3
        );
    }

    #[test]
    fn not_the_sphere() {
        let m = SeN::new(2).unwrap();
        let x = identity_t(2, &[1.0, 0.0]);
        let v = m.project(&x, &ambient_step(2, 0.4, &[0.0, 0.0]));
        let y = m.retract(&x, &v);
        let fro2: f64 = y.iter().map(|a| a * a).sum();
        assert!((fro2 - 1.0).abs() > 0.5, "must not be a unit sphere {y:?}");
        assert!(is_se(&y));
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![0.0, -1.0, 1.0, 0.0, 1.0, 0.0];
        let v = array![0.0, -0.5, 0.5, 0.0, 2.0, 0.0];
        assert!((inner(&u, &v) - 3.0).abs() < 1e-15);
        let want = (2.0 * (std::f64::consts::PI * std::f64::consts::PI + 1.0)).sqrt();
        assert!((typical_dist(2) - want).abs() < 1e-15);
        assert!((nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn wrong_dim_keeps_length() {
        let m = SeN::new(4).unwrap();
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(m.project(&x, &v).len(), 114);
        assert!(m.required_dim(20).is_ok());
        assert!(m.required_dim(12).is_err());
        assert!(m.required_dim(16).is_err());
        assert!(m.required_dim(114).is_err());
        assert!(SeN::new(1).is_err());
        assert!(SeN::new(0).is_err());
        assert!(side(6).is_some());
        assert!(side(12).is_some());
        assert!(side(1).is_none());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = SeN::new(4).unwrap();
        let x = identity_t(4, &[0.2, -0.1, 0.0, 0.3]);
        let y = m.retract(&x, &Array1::zeros(20));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_se(&y));
    }
}
