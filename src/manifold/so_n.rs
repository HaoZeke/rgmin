//! Special orthogonal group \(\mathrm{SO}(n)\) for \(n \ge 2\).
//!
//! manopt `rotationsfactory(n)` at `k = 1`. A point is an `n x n`
//! rotation, packed row-major as length `n^2` (same convention as
//! [`super::So3`]). Tangent vectors are the ambient embedding
//! \(R\Omega\) with \(\Omega^\top = -\Omega\), not the Lie-algebra
//! factor alone.
//!
//! Projection is \(R\,\mathrm{skew}(R^\top H)\). Retraction is
//! `qr_unique` of \(R + V\) (manopt `retr_qr`) with the last column
//! flipped if \(\det < 0\). Transport is projection at the arrival
//! point. Inner products go through [`crate::vecops`].
//!
//! [`super::So3`] stays the dedicated 9-vector token. This factory
//! does not reuse [`super::Sphere`] and does not pack a 3N cluster
//! as Stiefel. Isolated molecules use [`super::RigidQuotient`].

use ndarray::{Array1, ArrayView1};

use crate::vecops::{axpy, dot, nrm2};

use super::Manifold;

/// \(\mathrm{SO}(n)\), \(n \ge 2\). Packed row-major, length `n^2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoN {
    /// Matrix side. Must be `>= 2`.
    pub n: usize,
}

impl SoN {
    /// \(\mathrm{SO}(n)\) with \(n \ge 2\).
    pub fn new(n: usize) -> Result<Self, usize> {
        if n >= 2 {
            Ok(Self { n })
        } else {
            Err(n)
        }
    }

    /// Packed length `n^2`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(self.n)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= 2 && self.packed_len() == Some(len)
    }

    /// Row-major n-by-n rotation as a length-`n^2` token.
    pub fn pack(&self, mat: &[f64]) -> Option<Array1<f64>> {
        if !self.fits(mat.len()) {
            return None;
        }
        Some(Array1::from(mat.to_vec()))
    }

    /// Row-major storage of a packed point or tangent.
    pub fn unpack(&self, x: &Array1<f64>) -> Option<Vec<f64>> {
        if !self.fits(x.len()) {
            return None;
        }
        Some(x.to_vec())
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

/// Side length if `len` is `n^2` for some \(n \ge 2\).
pub fn side(len: usize) -> Option<usize> {
    if len < 4 {
        return None;
    }
    let n = (len as f64).sqrt().round() as usize;
    if n >= 2 && n.checked_mul(n) == Some(len) {
        Some(n)
    } else {
        None
    }
}

/// Split a length-n² ambient vector into (n, row-major entries).
pub fn unpack(x: &Array1<f64>) -> Option<(usize, Vec<f64>)> {
    let n = side(x.len())?;
    Some((n, x.iter().copied().collect()))
}

/// Flatten a row-major n-by-n matrix into the ambient vector.
pub fn pack(n: usize, a: Vec<f64>) -> Array1<f64> {
    Array1::from_shape_vec(n * n, a).unwrap()
}

/// Frobenius inner product. manopt `M.inner = d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    dot(u.view(), v.view())
}

/// manopt `M.typicaldist = pi*sqrt(n*k)` with `k = 1`.
pub fn typical_dist(n: usize) -> f64 {
    std::f64::consts::PI * (n as f64).sqrt()
}

/// `true` when the packed matrix is in \(\mathrm{SO}(n)\).
pub fn is_so(x: &Array1<f64>) -> bool {
    let Some((n, a)) = unpack(x) else {
        return false;
    };
    let Ok(m) = SoN::new(n) else {
        return false;
    };
    is_rotation(&m, &a)
}

fn is_rotation(m: &SoN, a: &[f64]) -> bool {
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

impl Manifold for SoN {
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
        let rt = self.transpose(xv);
        let rth = self.mul(&rt, hv);
        let omega = self.skew(&rth);
        pack(self.n, self.mul(xv, &omega))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (x.as_slice(), v.as_slice()) else {
            return x + v;
        };
        if !self.fits(xv.len()) || xv.len() != uv.len() {
            return x + v;
        }
        let mut y = xv.to_vec();
        for (yi, ui) in y.iter_mut().zip(uv.iter()) {
            *yi += *ui;
        }
        self.qr_pos(&mut y);
        pack(self.n, y)
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn identity(n: usize) -> Array1<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        pack(n, a)
    }

    fn skew_step(n: usize, scale: f64) -> Array1<f64> {
        let mut a = vec![0.0; n * n];
        if n >= 2 {
            a[1] = -scale;
            a[n] = scale;
        }
        if n >= 4 {
            a[2 * n + 3] = -0.5 * scale;
            a[3 * n + 2] = 0.5 * scale;
        }
        pack(n, a)
    }

    #[test]
    fn retract_stays_on_so2() {
        let m = SoN::new(2).unwrap();
        let x = identity(2);
        let v = m.project(&x, &skew_step(2, 0.3));
        let y = m.retract(&x, &v);
        assert!(is_so(&y), "left SO(2) {y:?}");
        assert!((y[0] * y[3] - y[1] * y[2] - 1.0).abs() < 1e-12);
        let fro2: f64 = y.iter().map(|a| a * a).sum();
        assert!((fro2 - 2.0).abs() < 1e-12, "not a rotation {y:?}");
    }

    #[test]
    fn retract_stays_on_so4() {
        let m = SoN::new(4).unwrap();
        let x = identity(4);
        let v = m.project(&x, &skew_step(4, 0.2));
        let y = m.retract(&x, &v);
        assert!(is_so(&y), "left SO(4) {y:?}");
        assert_eq!(y.len(), 16);
        assert!(det(4, y.as_slice().unwrap()) > 0.0);
    }

    #[test]
    fn project_is_horizontal() {
        let m = SoN::new(4).unwrap();
        let x = identity(4);
        let v = array![
            0.2, 0.1, -0.3, 0.4, 0.5, -0.2, 0.1, 0.0, 0.0, 0.3, -0.1, 0.2, -0.4, 0.0, 0.2, 0.1
        ];
        let z = m.project(&x, &v);
        let rt = m.transpose(x.as_slice().unwrap());
        let rth = m.mul(&rt, z.as_slice().unwrap());
        for i in 0..4 {
            for j in 0..4 {
                let s = rth[i * 4 + j] + rth[j * 4 + i];
                assert!(s.abs() < 1e-12, "R^T Z not skew [{i},{j}] = {s}");
            }
        }
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let m = SoN::new(2).unwrap();
        let x = identity(2);
        let v = m.project(&x, &skew_step(2, 0.25));
        let y = m.retract(&x, &v);
        let t = m.transport(&x, &y, &v);
        let p = m.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(is_so(&y));
    }

    #[test]
    fn pack_unpack_is_row_major() {
        let m = SoN::new(2).unwrap();
        let mat = [0.0, -1.0, 1.0, 0.0];
        let x = m.pack(&mat).unwrap();
        assert_eq!(x.len(), 4);
        assert_eq!(m.unpack(&x).unwrap(), mat);
        assert!(m.pack(&[1.0, 0.0, 0.0]).is_none());
        let (n, flat) = unpack(&x).unwrap();
        assert_eq!(n, 2);
        let y = pack(n, flat);
        for i in 0..4 {
            assert!((x[i] - y[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn so3_kind_is_not_this_factory() {
        let m = SoN::new(3).unwrap();
        let x = identity(3);
        let v = m.project(&x, &skew_step(3, 0.1));
        let y = m.retract(&x, &v);
        let z = super::super::So3.retract(&x, &v);
        for (a, b) in y.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12, "SoN(3) must match So3 geometry");
        }
        assert_ne!(
            crate::manifold::ManifoldKind::so_n(3),
            crate::manifold::ManifoldKind::So3
        );
        assert_ne!(
            crate::manifold::ManifoldKind::so_n(2),
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::so_n(2),
            crate::manifold::ManifoldKind::Stiefel
        );
    }

    #[test]
    fn not_the_sphere() {
        let m = SoN::new(2).unwrap();
        let x = identity(2);
        let v = m.project(&x, &skew_step(2, 0.4));
        let y = m.retract(&x, &v);
        let fro2: f64 = y.iter().map(|a| a * a).sum();
        assert!((fro2 - 1.0).abs() > 0.5, "must not be a unit sphere {y:?}");
        assert!(is_so(&y));
    }

    #[test]
    fn frobenius_inner_and_typical_dist() {
        let u = array![0.0, -1.0, 1.0, 0.0];
        let v = array![0.0, -0.5, 0.5, 0.0];
        assert!((inner(&u, &v) - 1.0).abs() < 1e-15);
        let want = std::f64::consts::PI * 2.0_f64.sqrt();
        assert!((typical_dist(2) - want).abs() < 1e-15);
        assert!((nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn wrong_dim_keeps_length() {
        let m = SoN::new(4).unwrap();
        let x = Array1::from_elem(12, 0.1);
        let v = Array1::from_elem(12, 0.01);
        let y = m.retract(&x, &v);
        assert_eq!(y.len(), 12);
        assert_eq!(m.project(&x, &v).len(), 12);
        assert!(m.required_dim(16).is_ok());
        assert!(m.required_dim(12).is_err());
        assert!(m.required_dim(9).is_err());
        assert!(m.required_dim(114).is_err());
        assert!(SoN::new(1).is_err());
        assert!(SoN::new(0).is_err());
        assert!(side(9).is_some());
        assert!(side(12).is_none());
        assert!(side(1).is_none());
    }

    #[test]
    fn zero_step_is_the_point() {
        let m = SoN::new(4).unwrap();
        let x = identity(4);
        let y = m.retract(&x, &Array1::zeros(16));
        assert!((&y - &x).mapv(f64::abs).sum() < 1e-15);
        assert!(is_so(&y));
    }
}
