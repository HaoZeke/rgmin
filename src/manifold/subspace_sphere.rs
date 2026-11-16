//! Unit sphere of a linear subspace, \(S(U) = \{x \in U : \|x\| = 1\}\).
//!
//! \(U \subset \mathbb{R}^n\) is given by a basis of itself
//! ([`SubspaceSphere::within`]) or of its orthogonal complement
//! ([`SubspaceSphere::excluding`]), whichever is smaller: a rotation
//! subspace spanned by a few modes, or the centre-of-mass directions and
//! fixed atoms a sampler removes. The rows are orthonormalized once, and
//! \(P_U\) costs one pass over them.
//!
//! The tangent space at \(x\) is \(\{v \in U : x^\top v = 0\}\), with the
//! projector \(P_x = (I - xx^\top) P_U\). The retraction is
//! \(P_U(x+v)/\|P_U(x+v)\|\), transport is projection at the arrival
//! point, and the Hessian is \(P_x(\nabla^2 \bar f\,v - (x^\top \nabla \bar
//! f)\,v)\) for any smooth extension \(\bar f\): \(U\) is flat, so only the
//! radial part of the gradient bends the sphere. \(S(U)\) is a great
//! sphere, totally geodesic in \(S^{n-1}\), so the exponential, logarithm,
//! distance and parallel transport are the sphere's.

use ndarray::{Array1, Array2, Axis};

use super::Manifold;
use super::sphere::{dist_unit, exp_unit, log_unit, transport_along_unit};
use crate::error::{Error, Result};

/// Unit sphere of a linear subspace in the ambient Euclidean metric.
#[derive(Clone, Debug, PartialEq)]
pub struct SubspaceSphere {
    /// Orthonormal rows spanning \(U\), or \(U^\perp\) when `excludes`.
    basis: Array2<f64>,
    excludes: bool,
}

// Basis entries are validated finite, never NaN.
impl Eq for SubspaceSphere {}

fn refuse(what: &'static str) -> Error {
    Error::SubspaceBasis { what }
}

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

/// Modified Gram–Schmidt with one reorthogonalization pass. A row whose
/// part outside the span of the earlier rows is below \(\sqrt{\varepsilon}\)
/// of its length is dependent.
fn orthonormal_rows(mut basis: Array2<f64>) -> Result<Array2<f64>> {
    if basis.ncols() == 0 {
        return Err(refuse("the ambient dimension must be positive"));
    }
    if basis.iter().any(|e| !e.is_finite()) {
        return Err(refuse("a basis entry is not finite"));
    }
    for i in 0..basis.nrows() {
        let (done, mut rest) = basis.view_mut().split_at(Axis(0), i);
        let mut row = rest.row_mut(0);
        let length = row.dot(&row).sqrt();
        for _ in 0..2 {
            for q in done.rows() {
                let c = q.dot(&row);
                row.scaled_add(-c, &q);
            }
        }
        let residual = row.dot(&row).sqrt();
        if !(length > 0.0 && residual > f64::EPSILON.sqrt() * length) {
            return Err(refuse("the basis rows are linearly dependent"));
        }
        row /= residual;
    }
    Ok(basis)
}

impl SubspaceSphere {
    /// Sphere of the span of the rows of `basis`, each of length `n`.
    ///
    /// Fails with [`Error::SubspaceBasis`] when `n` is zero, an entry is
    /// not finite, the rows are linearly dependent, or there is no row.
    pub fn within(basis: Array2<f64>) -> Result<Self> {
        let basis = orthonormal_rows(basis)?;
        if basis.nrows() == 0 {
            return Err(refuse("the subspace needs at least one direction"));
        }
        Ok(Self {
            basis,
            excludes: false,
        })
    }

    /// Sphere of the orthogonal complement of the rows of `basis`: the
    /// unit vectors with no component along an excluded direction. No row
    /// is the whole sphere \(S^{n-1}\).
    ///
    /// Fails with [`Error::SubspaceBasis`] when `n` is zero, an entry is
    /// not finite, the rows are linearly dependent, or they span all of
    /// \(\mathbb{R}^n\).
    pub fn excluding(basis: Array2<f64>) -> Result<Self> {
        let basis = orthonormal_rows(basis)?;
        if basis.nrows() >= basis.ncols() {
            return Err(refuse("the excluded directions leave no subspace"));
        }
        Ok(Self {
            basis,
            excludes: true,
        })
    }

    /// Ambient dimension `n`, the packed length of a point.
    pub fn ambient_dim(&self) -> usize {
        self.basis.ncols()
    }

    /// Dimension of \(U\). The sphere has one less.
    pub fn subspace_dim(&self) -> usize {
        if self.excludes {
            self.basis.ncols() - self.basis.nrows()
        } else {
            self.basis.nrows()
        }
    }

    /// Orthonormal rows spanning \(U\), or \(U^\perp\) when
    /// [`Self::excludes`].
    pub fn basis(&self) -> &Array2<f64> {
        &self.basis
    }

    /// `true` when [`Self::basis`] spans the excluded directions.
    pub fn excludes(&self) -> bool {
        self.excludes
    }

    /// \(P_U v\). A vector of another length comes back unchanged.
    pub fn project_subspace(&self, v: &Array1<f64>) -> Array1<f64> {
        if v.len() != self.ambient_dim() {
            return v.clone();
        }
        let along = self.basis.t().dot(&self.basis.dot(v));
        if self.excludes { v - &along } else { along }
    }

    /// The point of \(S(U)\) nearest to `x`, \(P_U x / \|P_U x\|\). `None`
    /// when `x` has another length or no component in \(U\).
    pub fn closest_point(&self, x: &Array1<f64>) -> Option<Array1<f64>> {
        if x.len() != self.ambient_dim() {
            return None;
        }
        let p = self.project_subspace(x);
        let norm = dot(&p, &p).sqrt();
        (norm > 1e-16).then(|| p / norm)
    }

    fn fits(&self, arrays: &[&Array1<f64>]) -> bool {
        arrays.iter().all(|a| a.len() == self.ambient_dim())
    }
}

impl Manifold for SubspaceSphere {
    fn required_dim(&self, n: usize) -> std::result::Result<(), usize> {
        let dim = self.ambient_dim();
        if n == dim { Ok(()) } else { Err(dim) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(&[x, v]) {
            return v.clone();
        }
        let mut p = self.project_subspace(v);
        let s = dot(x, &p);
        p.zip_mut_with(x, |pi, &xi| *pi -= s * xi);
        p
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if !self.fits(&[x, v]) {
            return x.clone();
        }
        self.closest_point(&(x + v))
            .or_else(|| self.closest_point(x))
            .unwrap_or_else(|| x.clone())
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
        if !self.fits(&[x, egrad, ehess, v]) {
            return None;
        }
        let normal = dot(x, egrad);
        Some(self.project(x, &(ehess - &(v * normal))))
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        self.fits(&[x, v]).then(|| exp_unit(x.view(), v.view()))
    }

    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        self.fits(&[x, y]).then(|| log_unit(x.view(), y.view()))
    }

    fn dist(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
        self.fits(&[x, y]).then(|| dist_unit(x.view(), y.view()))
    }

    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        self.fits(&[x, d, v])
            .then(|| transport_along_unit(x.view(), d.view(), v.view()))
    }

    fn typical_dist(&self, _n: usize) -> f64 {
        std::f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn excluding_e3() -> SubspaceSphere {
        SubspaceSphere::excluding(array![[0.0, 0.0, 2.0, 0.0]]).unwrap()
    }

    #[test]
    fn rows_are_orthonormalized() {
        let s = SubspaceSphere::within(array![[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]]).unwrap();
        let gram = s.basis().dot(&s.basis().t());
        let eye = Array2::<f64>::eye(2);
        assert!((&gram - &eye).iter().all(|e| e.abs() < 1e-15), "{gram:?}");
        assert_eq!(s.subspace_dim(), 2);
        assert_eq!(excluding_e3().subspace_dim(), 3);
    }

    #[test]
    fn bad_bases_are_refused() {
        let refused = |r: Result<SubspaceSphere>| matches!(r, Err(Error::SubspaceBasis { .. }));
        assert!(refused(SubspaceSphere::within(Array2::zeros((1, 0)))));
        assert!(refused(SubspaceSphere::within(Array2::zeros((0, 3)))));
        assert!(refused(SubspaceSphere::within(array![[1.0, f64::NAN]])));
        assert!(refused(SubspaceSphere::within(array![
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 0.0]
        ])));
        assert!(refused(SubspaceSphere::excluding(Array2::eye(2))));
        let whole = SubspaceSphere::excluding(Array2::zeros((0, 3))).unwrap();
        assert_eq!(whole.subspace_dim(), 3);
    }

    #[test]
    fn projection_and_retraction_stay_in_the_subspace() {
        let s = excluding_e3();
        let x = array![0.6, 0.0, 0.0, 0.8];
        let v = s.project(&x, &array![1.0, 2.0, 3.0, 4.0]);
        assert!(v[2].abs() < 1e-15 && dot(&x, &v).abs() < 1e-15, "{v:?}");
        let y = s.retract(&x, &v);
        assert!(y[2].abs() < 1e-15 && (dot(&y, &y) - 1.0).abs() < 1e-15);
        let back = s.retract(&x, &(-&x));
        assert!((&back - &x).iter().all(|e| e.abs() < 1e-15), "{back:?}");
    }

    #[test]
    fn the_hessian_bends_only_along_the_point() {
        let s = excluding_e3();
        let x = array![0.0, 1.0, 0.0, 0.0];
        let egrad = array![0.5, 3.0, 7.0, 0.0];
        let ehess = array![1.0, 2.0, 5.0, -1.0];
        let v = array![1.0, 0.0, 0.0, 0.0];
        let h = s.ehess2rhess(&x, &egrad, &ehess, &v).unwrap();
        assert_eq!(h, array![1.0 - 3.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn closest_point_needs_a_component_in_the_subspace() {
        let s = excluding_e3();
        assert_eq!(
            s.closest_point(&array![0.0, 3.0, 1.0, 4.0]),
            Some(array![0.0, 0.6, 0.0, 0.8])
        );
        assert_eq!(s.closest_point(&array![0.0, 0.0, 1.0, 0.0]), None);
    }
}
