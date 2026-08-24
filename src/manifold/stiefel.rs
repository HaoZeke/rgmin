//! Stiefel \(\mathrm{St}(n,p)\): orthonormal n-by-p frames.
//!
//! `p = 1` is manopt `stiefelfactory(n, 1)`: a length-\(n\) unit
//! vector. Projection, QR retraction, and transport match the
//! sphere. The inner product is the Frobenius / Euclidean dot
//! (`d1(:).'*d2(:)`). Typical distance is \(\sqrt{p k} = 1\) at
//! `k = 1` (not \(\pi\); that is `spherefactory`).
//!
//! `p > 1` is packed column-major (manopt `X(:)`), length `n*p`,
//! with \(X^\top X = I_p\). Projection is \(U - X\,\mathrm{sym}(X^\top U)\).
//! Retraction is thin QR with positive diagonal (manopt `retr_qr`).
//! Transport is projection at the arrival point.
//!
//! A 3N cluster is [`super::RigidQuotient`], not \(\mathrm{St}(3N, 1)\).
//! `p` is named on [`StiefelNp`]: length `n*p` does not name `p`.

use ndarray::{Array1, ArrayView1};

use crate::vecops::{self, axpy, dot, nrm2};

use super::{sphere::Sphere, Manifold};

/// Stiefel with \(p=1\), packed as a length-\(n\) unit vector.
/// manopt `stiefelfactory(n, 1)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stiefel;

/// Frobenius inner product. manopt `M.inner = d1(:).'*d2(:)`.
pub fn inner(u: &Array1<f64>, v: &Array1<f64>) -> f64 {
    vecops::dot(u.view(), v.view())
}

/// manopt `M.typicaldist = @() sqrt(p*k)` at `p = 1`, `k = 1`.
pub fn typical_dist() -> f64 {
    1.0
}

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

/// Stiefel \(\mathrm{St}(n,p)\) for `p > 1`. Packed column-major, length `n*p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StiefelNp {
    /// Rows of the frame.
    pub n: usize,
    /// Orthonormal columns. Must be `> 1`.
    pub p: usize,
}

impl StiefelNp {
    /// \(\mathrm{St}(n,p)\) with \(n \ge p \ge 2\).
    pub fn new(n: usize, p: usize) -> Result<Self, (usize, usize)> {
        if n >= p && p >= 2 {
            Ok(Self { n, p })
        } else {
            Err((n, p))
        }
    }

    /// Packed length `n*p`, or `None` on overflow.
    pub fn packed_len(self) -> Option<usize> {
        self.n.checked_mul(self.p)
    }

    fn fits(self, len: usize) -> bool {
        self.n >= self.p && self.p >= 2 && self.packed_len() == Some(len)
    }

    /// Column-major n-by-p frame as a length-`n*p` token (manopt `X(:)`).
    pub fn pack(&self, frame: &[f64]) -> Option<Array1<f64>> {
        if !self.fits(frame.len()) {
            return None;
        }
        Some(Array1::from(frame.to_vec()))
    }

    /// Column-major storage of a packed point or tangent.
    pub fn unpack(&self, x: &Array1<f64>) -> Option<Vec<f64>> {
        if !self.fits(x.len()) {
            return None;
        }
        Some(x.to_vec())
    }

    fn col<'a>(&self, a: &'a [f64], j: usize) -> ArrayView1<'a, f64> {
        ArrayView1::from(&a[j * self.n..(j + 1) * self.n])
    }

    fn write_col(&self, a: &mut [f64], j: usize, col: &Array1<f64>) {
        let sl = &mut a[j * self.n..(j + 1) * self.n];
        if let Some(src) = col.as_slice() {
            sl.copy_from_slice(src);
        } else {
            for (dst, src) in sl.iter_mut().zip(col.iter()) {
                *dst = *src;
            }
        }
    }

    fn xtu(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let mut s = vec![0.0; self.p * self.p];
        for a in 0..self.p {
            for b in 0..self.p {
                s[a + self.p * b] = dot(self.col(x, a), self.col(u, b));
            }
        }
        s
    }

    fn sym_inplace(s: &mut [f64], p: usize) {
        for a in 0..p {
            for b in 0..a {
                let v = 0.5 * (s[a + p * b] + s[b + p * a]);
                s[a + p * b] = v;
                s[b + p * a] = v;
            }
        }
    }

    /// Thin QR with positive diagonal (Gram-Schmidt columns).
    fn qr_unique(&self, y: &mut [f64]) {
        for j in 0..self.p {
            let mut v = self.col(y, j).to_owned();
            for a in 0..j {
                let qa = self.col(y, a);
                let r = dot(qa, v.view());
                axpy(-r, qa, &mut v);
            }
            let nrm = nrm2(v.view());
            if nrm > 1e-16 {
                v.mapv_inplace(|t| t / nrm);
            }
            self.write_col(y, j, &v);
        }
    }
}

impl Manifold for StiefelNp {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self.packed_len() {
            Some(want) if self.n >= self.p && self.p >= 2 && n == want => Ok(()),
            Some(want) => Err(want),
            None => Err(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (x.as_slice(), v.as_slice()) else {
            return v.clone();
        };
        if !self.fits(xv.len()) || xv.len() != uv.len() {
            return v.clone();
        }
        let mut xtu = self.xtu(xv, uv);
        Self::sym_inplace(&mut xtu, self.p);
        let mut out = uv.to_vec();
        for j in 0..self.p {
            let mut col = self.col(&out, j).to_owned();
            for a in 0..self.p {
                axpy(-xtu[a + self.p * j], self.col(xv, a), &mut col);
            }
            self.write_col(&mut out, j, &col);
        }
        Array1::from(out)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (x.as_slice(), v.as_slice()) else {
            return x + v;
        };
        if !self.fits(xv.len()) || xv.len() != uv.len() {
            return x + v;
        }
        let mut y = Array1::from(xv.to_vec());
        axpy(1.0, ArrayView1::from(uv), &mut y);
        let Some(ys) = y.as_slice_mut() else {
            return x + v;
        };
        self.qr_unique(ys);
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

    fn frame_4x2() -> Array1<f64> {
        // Columns (1,0,0,0) and (0,1,0,0), column-major.
        array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    #[test]
    fn p1_matches_sphere_in_all_three_operations() {
        let x = array![0.6, 0.8, 0.0];
        let v = array![0.1, -0.2, 0.5];
        let y = array![0.0, 1.0, 0.0];
        assert_eq!(Stiefel.project(&x, &v), Sphere.project(&x, &v));
        assert_eq!(Stiefel.retract(&x, &v), Sphere.retract(&x, &v));
        assert_eq!(Stiefel.transport(&x, &y, &v), Sphere.transport(&x, &y, &v));
        let r = Stiefel.egrad2rgrad(&x, &v);
        assert_eq!(r, Stiefel.project(&x, &v));
        assert_eq!(r, Sphere.egrad2rgrad(&x, &v));
    }

    #[test]
    fn p1_retract_stays_on_the_set() {
        let x = array![0.0, 1.0, 0.0];
        let v = array![0.1, 0.0, -0.2];
        let y = Stiefel.retract(&x, &v);
        assert!(
            (vecops::nrm2(y.view()) - 1.0).abs() < 1e-14,
            "left St(n,1) {y:?}"
        );
        let z = Stiefel.retract(&x, &Array1::zeros(3));
        assert!((vecops::nrm2(z.view()) - 1.0).abs() < 1e-15);
        for i in 0..3 {
            assert!((z[i] - x[i]).abs() < 1e-15, "zero step {z:?} != {x:?}");
        }
        assert!(Stiefel.required_dim(y.len()).is_ok());
    }

    #[test]
    fn p1_project_is_tangent() {
        let x = array![1.0, 0.0, 0.0];
        let v = array![2.0, 3.0, 4.0];
        let t = Stiefel.project(&x, &v);
        assert!(inner(&x, &t).abs() < 1e-15, "x·t = {}", inner(&x, &t));
        assert!((t[1] - 3.0).abs() < 1e-15);
        assert!((t[2] - 4.0).abs() < 1e-15);
        assert!(t[0].abs() < 1e-15);
        let tt = Stiefel.project(&x, &t);
        for i in 0..3 {
            assert!((tt[i] - t[i]).abs() < 1e-15, "P^2 != P {tt:?} != {t:?}");
        }
    }

    #[test]
    fn p1_transport_is_projection_at_arrival() {
        let x = array![1.0, 0.0, 0.0];
        let v = Stiefel.project(&x, &array![0.0, 0.3, -0.4]);
        let y = Stiefel.retract(&x, &v);
        let w = array![0.2, -0.1, 0.5];
        let t = Stiefel.transport(&x, &y, &w);
        assert!(
            inner(&y, &t).abs() < 1e-14,
            "arrival x·T(v) = {} at {y:?} t={t:?}",
            inner(&y, &t)
        );
        let p = Stiefel.project(&y, &w);
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
        assert!((typical_dist() - 1.0).abs() < 1e-15);
        assert!((typical_dist() - std::f64::consts::PI).abs() > 1.0);
        assert!((vecops::nrm2(u.view()) - inner(&u, &u).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn p_greater_than_one_is_stiefel_p_not_this_token() {
        use crate::manifold::ManifoldKind;
        assert_eq!(
            ManifoldKind::stiefel(4, 2),
            ManifoldKind::StiefelP { n: 4, p: 2 }
        );
        assert_ne!(ManifoldKind::stiefel(4, 2), ManifoldKind::Stiefel);
        assert_eq!(ManifoldKind::stiefel(5, 1), ManifoldKind::Stiefel);
        assert_eq!(ManifoldKind::stiefel(5, 1).stiefel_p(), 1);
        assert_eq!(ManifoldKind::stiefel(4, 2).stiefel_p(), 2);
        assert!(StiefelNp::new(4, 1).is_err());
        assert!(StiefelNp::new(4, 2).is_ok());
    }

    #[test]
    fn a_3n_cluster_is_not_stiefel() {
        use crate::manifold::ManifoldKind;
        assert_ne!(ManifoldKind::Stiefel, ManifoldKind::RigidQuotient);
        assert_ne!(ManifoldKind::Stiefel, ManifoldKind::MwRigid);
        assert_ne!(ManifoldKind::stiefel(114, 1), ManifoldKind::RigidQuotient);
        let packed = ManifoldKind::stiefel(38, 3);
        assert_eq!(packed, ManifoldKind::StiefelP { n: 38, p: 3 });
        assert_ne!(packed, ManifoldKind::Stiefel);
        assert_ne!(packed, ManifoldKind::RigidQuotient);
        assert!(packed.required_dim(114).is_ok());
        assert!(ManifoldKind::RigidQuotient.required_dim(114).is_ok());
        assert_ne!(packed.as_str(), ManifoldKind::RigidQuotient.as_str());
    }

    #[test]
    fn project_is_horizontal() {
        let st = StiefelNp::new(4, 2).unwrap();
        let x = frame_4x2();
        let v = array![0.2, 0.1, 0.3, -0.4, 0.5, -0.2, 0.1, 0.7];
        let z = st.project(&x, &v);
        let xtz = st.xtu(x.as_slice().unwrap(), z.as_slice().unwrap());
        for a in 0..2 {
            for b in 0..2 {
                let s = xtz[a + 2 * b] + xtz[b + 2 * a];
                assert!(s.abs() < 1e-12, "X^T Z + Z^T X [{a},{b}] = {s}");
            }
        }
    }

    #[test]
    fn retract_stays_on_stiefel() {
        let st = StiefelNp::new(4, 2).unwrap();
        let x = frame_4x2();
        let v = st.project(&x, &array![0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2]);
        let y = st.retract(&x, &v);
        let yty = st.xtu(y.as_slice().unwrap(), y.as_slice().unwrap());
        for a in 0..2 {
            for b in 0..2 {
                let got = yty[a + 2 * b];
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((got - want).abs() < 1e-12, "Y^T Y[{a},{b}]={got}");
            }
        }
    }

    #[test]
    fn transport_is_projection_at_arrival() {
        let st = StiefelNp::new(4, 2).unwrap();
        let x = frame_4x2();
        let v = st.project(&x, &array![0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2]);
        let y = st.retract(&x, &v);
        let t = st.transport(&x, &y, &v);
        let p = st.project(&y, &v);
        for (a, b) in t.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        let ytt = st.xtu(y.as_slice().unwrap(), t.as_slice().unwrap());
        for a in 0..2 {
            for b in 0..2 {
                let s = ytt[a + 2 * b] + ytt[b + 2 * a];
                assert!(s.abs() < 1e-12, "arrival not tangent [{a},{b}] = {s}");
            }
        }
    }

    #[test]
    fn pack_unpack_is_column_major() {
        let st = StiefelNp::new(3, 2).unwrap();
        let frame = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let x = st.pack(&frame).unwrap();
        assert_eq!(x.len(), 6);
        assert_eq!(st.unpack(&x).unwrap(), frame);
        assert!(st.pack(&[1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn p_greater_than_one_rejects_a_3n_cluster_length() {
        let st = StiefelNp::new(4, 2).unwrap();
        assert!(st.required_dim(8).is_ok());
        assert!(st.required_dim(12).is_err());
        assert!(st.required_dim(114).is_err());
        assert!(Stiefel.required_dim(114).is_ok());
        assert!(StiefelNp::new(3, 4).is_err());
        assert!(StiefelNp::new(4, 1).is_err());
    }

    #[test]
    fn wrong_dim_keeps_length() {
        let st = StiefelNp::new(4, 2).unwrap();
        let x = Array1::from_elem(12, 0.1);
        let v = Array1::from_elem(12, 0.01);
        let y = st.retract(&x, &v);
        assert_eq!(y.len(), 12);
        assert_eq!(st.project(&x, &v).len(), 12);
    }
}
