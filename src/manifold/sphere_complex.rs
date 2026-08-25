//! Complex unit sphere \(\{z \in \mathbb{C}^n : z^* z = 1\}\).
//!
//! manopt `spherecomplexfactory`. Packed interleaved `(re, im)`,
//! length `2 n`. The real inner product \(\mathrm{Re}(x^* u)\) is
//! the Euclidean product on the packed vector, so projection and
//! retraction are those of the real sphere in \(\mathbb{R}^{2n}\).
//! This token is not [`super::Sphere`] (real \(S^{n-1}\)) and not
//! [`super::ComplexCircle`] (product of \(S^1\)). Reserved tokens
//! 7-10 stay unused.

use ndarray::{Array1, ArrayView1};

use super::{Manifold, Sphere};

/// Unit sphere in \(\mathbb{C}^n\), packed as `2 n` reals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereComplex {
    /// Complex dimension. Packed length is `2 n`.
    pub n: usize,
}

impl SphereComplex {
    /// Interleaved pack of real and imaginary parts.
    pub fn pack(re: ArrayView1<f64>, im: ArrayView1<f64>) -> Array1<f64> {
        let n = re.len().min(im.len());
        let mut out = Array1::zeros(2 * n);
        for k in 0..n {
            out[2 * k] = re[k];
            out[2 * k + 1] = im[k];
        }
        out
    }

    /// Split a packed vector. `None` if the length is odd.
    pub fn unpack(x: &Array1<f64>) -> Option<(Array1<f64>, Array1<f64>)> {
        if x.len() % 2 != 0 {
            return None;
        }
        let n = x.len() / 2;
        let mut re = Array1::zeros(n);
        let mut im = Array1::zeros(n);
        for k in 0..n {
            re[k] = x[2 * k];
            im[k] = x[2 * k + 1];
        }
        Some((re, im))
    }
}

impl Manifold for SphereComplex {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.n >= 1 && n == 2 * self.n {
            Ok(())
        } else {
            Err(2 * self.n.max(1))
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn retract_stays_on_the_complex_sphere() {
        let m = SphereComplex { n: 2 };
        let x = array![1.0, 0.0, 0.0, 0.0];
        let v = array![0.0, 0.1, 0.2, -0.1];
        let y = m.retract(&x, &v);
        let n2: f64 = y.iter().map(|yi| yi * yi).sum();
        assert!((n2.sqrt() - 1.0).abs() < 1e-14, "norm {n2}");
    }

    #[test]
    fn project_is_hermitian_tangent() {
        let m = SphereComplex { n: 2 };
        let x = array![0.6, 0.8, 0.0, 0.0];
        let v = array![1.0, 2.0, 3.0, 4.0];
        let t = m.project(&x, &v);
        let ip: f64 = x.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
        assert!(ip.abs() < 1e-14, "Re(x^* t) = {ip}");
    }

    #[test]
    fn required_dim_is_twice_complex_n() {
        let m = SphereComplex { n: 3 };
        assert_eq!(m.required_dim(6), Ok(()));
        assert_eq!(m.required_dim(3), Err(6));
    }

    #[test]
    fn not_the_real_sphere_token() {
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::Sphere
        );
        assert_ne!(
            crate::manifold::ManifoldKind::SphereComplex { n: 2 },
            crate::manifold::ManifoldKind::ComplexCircle { n: 2 }
        );
    }
}
