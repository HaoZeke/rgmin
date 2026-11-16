//! SE(3): row-major SO(3) (9) then translation (3). Length 12.
//!
//! Dimension is exactly 12. A 3N cluster is
//! [`super::RigidQuotient`], not a 12-vector prefix.

use ndarray::{Array1, s};

use super::{Manifold, so3::So3};

/// Rigid motions. Rotation block uses [`So3`]; translation is Euclidean.
#[derive(Clone, Copy, Debug, Default)]
pub struct Se3;

impl Manifold for Se3 {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == 12 { Ok(()) } else { Err(12) }
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
        let mut y = x + v;
        for i in 0..9 {
            y[i] = yr[i];
        }
        y
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }

    fn ehess2rhess(
        &self,
        x: &Array1<f64>,
        egrad: &Array1<f64>,
        ehess: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        if [x, egrad, ehess, v].iter().any(|a| a.len() != 12) {
            return None;
        }
        let rot = So3.ehess2rhess(&rot(x), &rot(egrad), &rot(ehess), &rot(v))?;
        Some(join(&rot, &ehess.slice(s![9..12]).to_owned()))
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        if x.len() != 12 || v.len() != 12 {
            return None;
        }
        let r = So3.exp(&rot(x), &rot(v))?;
        Some(join(&r, &(&trans(x) + &trans(v))))
    }

    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        if x.len() != 12 || y.len() != 12 {
            return None;
        }
        let r = So3.log(&rot(x), &rot(y))?;
        Some(join(&r, &(&trans(y) - &trans(x))))
    }

    fn dist(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
        if x.len() != 12 || y.len() != 12 {
            return None;
        }
        let dr = So3.dist(&rot(x), &rot(y))?;
        let dt = &trans(y) - &trans(x);
        Some((dr * dr + dt.dot(&dt)).sqrt())
    }

    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        if x.len() != 12 || d.len() != 12 || v.len() != 12 {
            return None;
        }
        let r = So3.parallel_transport_along(&rot(x), &rot(d), &rot(v))?;
        Some(join(&r, &trans(v)))
    }

    fn typical_dist(&self, _n: usize) -> f64 {
        let r = So3.typical_dist(9);
        (r * r + 3.0).sqrt()
    }
}

fn rot(a: &Array1<f64>) -> Array1<f64> {
    a.slice(s![0..9]).to_owned()
}

fn trans(a: &Array1<f64>) -> Array1<f64> {
    a.slice(s![9..12]).to_owned()
}

fn join(r: &Array1<f64>, t: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(12);
    out.slice_mut(s![0..9]).assign(r);
    out.slice_mut(s![9..12]).assign(t);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn eye_t(tx: f64, ty: f64, tz: f64) -> Array1<f64> {
        array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, tx, ty, tz]
    }

    fn rot_block(x: &Array1<f64>) -> [[f64; 3]; 3] {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = x[3 * i + j];
            }
        }
        r
    }

    #[test]
    fn translation_is_euclidean() {
        let x = eye_t(1.0, 2.0, 3.0);
        let v = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.1, 0.2];
        let y = Se3.retract(&x, &v);
        assert!((y[9] - 1.4).abs() < 1e-15);
        assert!((y[10] - 1.9).abs() < 1e-15);
        assert!((y[11] - 3.2).abs() < 1e-15);
        let r = rot_block(&y);
        for (i, row) in r.iter().enumerate() {
            for (j, &rij) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rij - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn rotation_block_stays_orthogonal() {
        let x = eye_t(0.0, 0.0, 0.0);
        let mut v = Array1::zeros(12);
        v[1] = -0.2;
        v[3] = 0.2;
        v[9] = 1.0;
        let y = Se3.retract(&x, &v);
        let r = rot_block(&y);
        let mut rtr = [[0.0; 3]; 3];
        for (i, row) in rtr.iter_mut().enumerate() {
            for (j, rij) in row.iter_mut().enumerate() {
                *rij = r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - want).abs() < 1e-12, "{rtr:?}");
            }
        }
        assert!((y[9] - 1.0).abs() < 1e-15);
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
        assert!(Se3.required_dim(114).is_err());
        assert!(Se3.required_dim(12).is_ok());
    }
}
