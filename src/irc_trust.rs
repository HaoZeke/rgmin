//! Sella `IRCTrustRegion` / Gonzalez--Schlegel MW sphere.
//!
//! Inner IRC step: \(\|(s + d_1)\odot\sqrt{m}\| = dx\).
//! \(d_1\) is the accumulated displacement from the last accepted
//! point. This is not [`crate::manifold::Sphere`] (unit \(S^{n-1}\)
//! about the origin).

use ndarray::Array1;

use crate::vecops::{axpy, div_assign_floor, mul_assign, nrm2, scale};

/// Per-atom masses (length N) to a 3N \(\sqrt{m}\) weight.
pub fn sqrt_masses_3n(masses: &[f64]) -> Array1<f64> {
    let mut out = Array1::zeros(masses.len() * 3);
    for (i, &m) in masses.iter().enumerate() {
        let s = m.max(0.0).sqrt();
        out[3 * i] = s;
        out[3 * i + 1] = s;
        out[3 * i + 2] = s;
    }
    out
}

/// Restricted step onto \(\|(s+d_1)\odot\sqrt{m}\| = dx\).
#[derive(Clone, Debug)]
pub struct IrcTrust {
    /// Displacement already taken from the last accepted IRC point.
    pub d1: Array1<f64>,
    /// Repeated \(\sqrt{m}\) weights, length 3N.
    pub sqrtm: Array1<f64>,
    /// Mass-weighted sphere radius.
    pub dx: f64,
}

impl IrcTrust {
    /// Build from per-atom masses (length N).
    pub fn from_atom_masses(d1: Array1<f64>, masses: &[f64], dx: f64) -> Self {
        Self {
            d1,
            sqrtm: sqrt_masses_3n(masses),
            dx: dx.max(0.0),
        }
    }

    /// \(\|(s + d_1)\odot\sqrt{m}\|\).
    pub fn cons(&self, s: &Array1<f64>) -> f64 {
        let n = s.len().min(self.d1.len()).min(self.sqrtm.len());
        if n == 0 {
            return 0.0;
        }
        let mut w = s.slice(ndarray::s![..n]).to_owned();
        axpy(1.0, self.d1.slice(ndarray::s![..n]), &mut w);
        mul_assign(self.sqrtm.slice(ndarray::s![..n]), &mut w);
        nrm2(w.view())
    }

    /// Radial projection of `s` onto the MW sphere of radius `dx`.
    ///
    /// Weighted add / scale / divide go through [`crate::vecops`].
    /// This path does not assemble a Hessian and does not call ELPA.
    pub fn project(&self, s: &Array1<f64>) -> Array1<f64> {
        let n = s.len().min(self.d1.len()).min(self.sqrtm.len());
        let mut out = s.clone();
        if n == 0 {
            return out;
        }
        let d1 = self.d1.slice(ndarray::s![..n]);
        let sm = self.sqrtm.slice(ndarray::s![..n]);
        let mut w = s.slice(ndarray::s![..n]).to_owned();
        axpy(1.0, d1, &mut w);
        mul_assign(sm, &mut w);
        let norm = nrm2(w.view());
        if norm <= 1e-16 {
            let wnorm = nrm2(sm);
            if wnorm <= 1e-16 || self.dx == 0.0 {
                let mut back = d1.to_owned();
                scale(-1.0, &mut back);
                out.slice_mut(ndarray::s![..n]).assign(&back);
                return out;
            }
            let mut back = d1.to_owned();
            scale(-1.0, &mut back);
            back[0] = self.dx / self.sqrtm[0].max(1e-16) - self.d1[0];
            out.slice_mut(ndarray::s![..n]).assign(&back);
            return out;
        }
        scale(self.dx / norm, &mut w);
        div_assign_floor(sm, &mut w, 1e-16);
        axpy(-1.0, d1, &mut w);
        out.slice_mut(ndarray::s![..n]).assign(&w);
        out
    }

    /// True when `cons(s)` sits on `dx` to `tol`.
    pub fn on_bound(&self, s: &Array1<f64>, tol: f64) -> bool {
        (self.cons(s) - self.dx).abs() <= tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn unequal_masses_sit_on_the_mw_sphere() {
        let masses = [1.0, 16.0];
        let d1 = array![0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let tr = IrcTrust::from_atom_masses(d1, &masses, 0.2);
        let s = array![0.5, 0.1, -0.2, 0.3, 0.0, 0.4];
        let p = tr.project(&s);
        assert!(
            tr.on_bound(&p, 1e-12),
            "cons={} dx={}",
            tr.cons(&p),
            tr.dx
        );
    }

    #[test]
    fn already_on_sphere_is_a_fixed_point() {
        let masses = [12.0, 1.0];
        let d1 = Array1::zeros(6);
        let tr = IrcTrust::from_atom_masses(d1, &masses, 0.15);
        let mut s = array![0.15, 0.0, 0.0, 0.0, 0.0, 0.0];
        s = tr.project(&s);
        let again = tr.project(&s);
        for (a, b) in s.iter().zip(again.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn inner_project_has_no_elpa_and_uses_vecops() {
        let src = include_str!("irc_trust.rs");
        let impl_only = src.split("#[cfg(test)]").next().expect("impl");
        for line in impl_only.lines() {
            let t = line.trim();
            if t.starts_with("//") || t.starts_with("///") {
                continue;
            }
            assert!(!t.to_ascii_lowercase().contains("elpa"), "ELPA in {t}");
            assert!(!t.contains("lowest_mode_dense"), "dense eigen in {t}");
        }
        assert!(impl_only.contains("axpy"));
        assert!(impl_only.contains("mul_assign"));
        assert!(impl_only.contains("nrm2"));
        assert!(impl_only.contains("scale"));
        assert!(impl_only.contains("div_assign_floor"));
    }

    #[cfg(feature = "par")]
    #[test]
    fn par_project_sits_on_the_mw_sphere() {
        let masses = [1.0, 12.0, 16.0];
        let d1 = Array1::zeros(9);
        let tr = IrcTrust::from_atom_masses(d1, &masses, 0.25);
        let s = array![0.4, -0.1, 0.2, 0.0, 0.3, -0.2, 0.1, 0.0, 0.05];
        let p = tr.project(&s);
        assert!(
            tr.on_bound(&p, 1e-12),
            "par cons={} dx={}",
            tr.cons(&p),
            tr.dx
        );
    }
}
