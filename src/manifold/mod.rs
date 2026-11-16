//! Embedded Riemannian manifolds (manopt_cpp / ROPTLIB waist).
//!
//! Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
//! Manifolds*, <https://doi.org/10.1515/9781400830244>.
//! Boumal, *An Introduction to Optimization on Smooth Manifolds*,
//! <https://doi.org/10.1017/9781009166164>.
//! manopt_cpp: `proj`, `retr`, `transp` on an embedded Euclidean vector.
//!
//! Isolated molecules and clusters use [`ManifoldKind::RigidQuotient`]
//! (Sella Cartesian `fix_translation` / `fix_rotation`,
//! \(R^{3N}/\mathrm{SE}(3)\)) or [`ManifoldKind::MwRigid`] (Page–McIver
//! mass-weighted Eckart, the IRC metric). Sphere / SO(3)-9 / SE(3)-12
//! / Symmetric-n² / SPD-n² / ComplexCircle-2n / EuclideanComplex-2n
//! / Constant-n / Positive-n / CenteredMatrix-mn / Oblique-nm are matrix-manifold embeddings, not a 3N cluster.
//! [`ManifoldKind::Symmetric`] is manopt `symmetricfactory`.
//! [`ManifoldKind::SkewSymmetric`] is manopt `skewsymmetricfactory`.
//! [`ManifoldKind::ComplexCircle`] is manopt `complexcirclefactory`.
//! [`ManifoldKind::EuclideanComplex`] is manopt `euclideancomplexfactory`.
//! [`ManifoldKind::Constant`] is manopt `constantfactory`.
//! [`ManifoldKind::MultinomialDoublyStochastic`] is manopt
//! `multinomialdoublystochasticfactory`.
//! [`ManifoldKind::MultinomialSymmetric`] is manopt
//! `multinomialsymmetricfactory`.
//! [`ManifoldKind::SphereComplex`] is manopt `spherecomplexfactory`.
//! [`ManifoldKind::Positive`] is manopt `positivefactory`.
//! [`ManifoldKind::CenteredMatrix`] is manopt `centeredmatrixfactory`.

use ndarray::Array1;

/// Bind the geometry a [`ManifoldKind`] names to `$f` and evaluate `$body`.
macro_rules! with_kind {
    ($kind:expr, |$f:ident| $body:expr) => {
        match $kind {
            ManifoldKind::Euclidean => {
                let $f = Euclidean;
                $body
            }
            ManifoldKind::Sphere => {
                let $f = Sphere;
                $body
            }
            ManifoldKind::So3 => {
                let $f = So3;
                $body
            }
            ManifoldKind::Stiefel => {
                let $f = Stiefel;
                $body
            }
            ManifoldKind::Se3 => {
                let $f = Se3;
                $body
            }
            ManifoldKind::RigidQuotient => {
                let $f = RigidQuotient;
                $body
            }
            ManifoldKind::MwRigid => {
                let $f = MwRigid;
                $body
            }
            ManifoldKind::Oblique { n, m } => {
                let $f = Oblique { n: *n, m: *m };
                $body
            }
            ManifoldKind::Multinomial => {
                let $f = Multinomial;
                $body
            }
            ManifoldKind::StiefelP { n, p } => {
                let $f = StiefelNp { n: *n, p: *p };
                $body
            }
            ManifoldKind::Spd => {
                let $f = Spd;
                $body
            }
            ManifoldKind::Symmetric => {
                let $f = Symmetric;
                $body
            }
            ManifoldKind::SkewSymmetric => {
                let $f = SkewSymmetric;
                $body
            }
            ManifoldKind::ComplexCircle { n } => {
                let $f = ComplexCircle { n: *n };
                $body
            }
            ManifoldKind::EuclideanComplex { n } => {
                let $f = EuclideanComplex { n: *n };
                $body
            }
            ManifoldKind::Constant { n } => {
                let $f = Constant { n: *n };
                $body
            }
            ManifoldKind::MultinomialDoublyStochastic { n } => {
                let $f = MultinomialDoublyStochastic { n: *n };
                $body
            }
            ManifoldKind::MultinomialSymmetric { n } => {
                let $f = MultinomialSymmetric { n: *n };
                $body
            }
            ManifoldKind::SphereComplex { n } => {
                let $f = SphereComplex { n: *n };
                $body
            }
            ManifoldKind::Positive { n } => {
                let $f = Positive { n: *n };
                $body
            }
            ManifoldKind::CenteredMatrix { m, n, rows } => {
                let $f = CenteredMatrix {
                    m: *m,
                    n: *n,
                    rows: *rows,
                };
                $body
            }
        }
    };
}

mod centered;
mod complex_circle;
mod constant;
mod euclidean;
mod euclidean_complex;
mod multinomial;
mod multinomial_ds;
mod multinomial_sym;
mod mw_rigid;
mod oblique;
mod positive;
mod rigid_quotient;
mod se3;
mod skewsymmetric;
mod so3;
mod spd;
mod sphere;
mod sphere_complex;
mod stiefel;
mod symmetric;

pub use centered::{
    CenteredMatrix, inner as inner_centered, is_centered, pack as pack_centered,
    typical_dist as typical_dist_centered, unpack as unpack_centered,
};
pub use complex_circle::ComplexCircle;
pub use constant::{
    Constant, inner as inner_const, is_constant, typical_dist as typical_dist_const,
};
pub use euclidean::Euclidean;
pub use euclidean_complex::{
    EuclideanComplex, inner as inner_cplx, is_euclidean_complex, typical_dist as typical_dist_cplx,
};
pub use multinomial::Multinomial;
pub use multinomial_ds::{
    MultinomialDoublyStochastic, inner as inner_ds, is_doubly_stochastic, pack as pack_ds,
    side as side_ds, typical_dist as typical_dist_ds, unpack as unpack_ds,
};
pub use multinomial_sym::{
    MultinomialSymmetric, inner as inner_msym, is_symmetric_doubly_stochastic, pack as pack_msym,
    side as side_msym, typical_dist as typical_dist_msym, unpack as unpack_msym,
};
pub use mw_rigid::MwRigid;
pub use oblique::Oblique;
pub use positive::{Positive, inner as inner_pos, is_positive, typical_dist as typical_dist_pos};
pub use rigid_quotient::RigidQuotient;
pub use se3::Se3;
pub use skewsymmetric::{
    SkewSymmetric, inner as inner_skew, is_skewsymmetric, pack as pack_skew, side as side_skew,
    typical_dist as typical_dist_skew, unpack as unpack_skew,
};
pub use so3::So3;
pub use spd::{Spd, is_spd, pack as pack_spd, side as side_spd, unpack as unpack_spd};
pub use sphere::Sphere;
pub use sphere_complex::{
    SphereComplex, inner as inner_scplx, is_sphere_complex, typical_dist as typical_dist_scplx,
};
pub use stiefel::{Stiefel, StiefelNp};
pub use symmetric::{
    Symmetric, inner as inner_sym, is_symmetric, pack as pack_sym, side as side_sym,
    typical_dist as typical_dist_sym, unpack as unpack_sym,
};

/// Which embedded geometry a session retracts onto.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ManifoldKind {
    /// Ambient Euclidean. Today's path.
    #[default]
    Euclidean,
    /// Unit sphere \(S^{n-1}\).
    Sphere,
    /// Rotation matrices SO(3), 9-vector row-major.
    So3,
    /// Stiefel \(\mathrm{St}(n,1)\). A single orthonormal column.
    /// `p > 1` is [`Self::StiefelP`]; a 3N cluster is [`Self::RigidQuotient`].
    Stiefel,
    /// Rigid motions SE(3): 3x3 row-major then translation (12).
    Se3,
    /// Isolated-molecule shape space \(R^{3N}/\mathrm{SE}(3)\).
    /// Sella Cartesian + `fix_translation` + `fix_rotation`.
    RigidQuotient,
    /// Mass-weighted Eckart: Sella IRC / Page–McIver metric on
    /// the same quotient. Masses from [`crate::Solver::set_masses`].
    MwRigid,
    /// Product of `m` unit spheres in \(\mathbb{R}^n\). Packed
    /// column-major, length `n*m`. manopt `obliquefactory`.
    Oblique {
        /// Ambient dimension of each sphere (column length).
        n: usize,
        /// Number of unit-norm columns.
        m: usize,
    },
    /// Simplex \(\{x > 0,\ \sum x = 1\}\) with the Fisher metric.
    /// manopt `multinomialfactory` at \(m = 1\). Length \(n \ge 2\).
    Multinomial,
    /// Stiefel \(\mathrm{St}(n,p)\) for `p > 1`, packed column-major
    /// length `n*p`. Construct with [`ManifoldKind::stiefel`].
    StiefelP {
        /// Ambient dimension. Rows of the frame.
        n: usize,
        /// Orthonormal columns. Must be `> 1`.
        p: usize,
    },
    /// Symmetric positive definite n-by-n, row-major n².
    /// manopt `sympositivedefinitefactory` (affine-invariant).
    Spd,
    /// Real symmetric n-by-n, row-major n².
    /// manopt `symmetricfactory` (Frobenius / Euclidean subspace).
    Symmetric,
    /// Real skew-symmetric n-by-n, row-major n², `n >= 2`.
    /// manopt `skewsymmetricfactory` (Frobenius / Euclidean subspace).
    SkewSymmetric,
    /// Product of unit-modulus complex numbers \((S^1)^n\).
    /// Packed interleaved `(re, im)`, length `2 n`.
    /// manopt `complexcirclefactory(n)`. Not the sphere.
    ComplexCircle {
        /// Number of unit-modulus complex entries.
        n: usize,
    },
    /// Complex Euclidean \(\mathbb{C}^n\). Packed interleaved
    /// `(re, im)`, length `2 n`. manopt `euclideancomplexfactory(n)`.
    /// Not the sphere and not \((S^1)^n\).
    EuclideanComplex {
        /// Number of complex entries.
        n: usize,
    },
    /// Singleton \(\{A\}\) of packed length `n`. manopt
    /// `constantfactory`. Projection and transport are zero;
    /// retraction is the fixed point. Not the sphere.
    Constant {
        /// Packed length of the singleton.
        n: usize,
    },
    /// Doubly-stochastic n-by-n, row-major n², `n >= 2`.
    /// manopt `multinomialdoublystochasticfactory`.
    /// Not the reserved SPD / grassmann / hyperbolic tokens.
    MultinomialDoublyStochastic {
        /// Side length.
        n: usize,
    },
    /// Symmetric doubly-stochastic n-by-n, row-major n², `n >= 2`.
    /// manopt `multinomialsymmetricfactory`.
    MultinomialSymmetric {
        /// Side length.
        n: usize,
    },
    /// Complex unit sphere in \(\mathbb{C}^n\), packed `2 n`.
    /// manopt `spherecomplexfactory`. Not the real sphere.
    SphereComplex {
        /// Complex dimension.
        n: usize,
    },
    /// Strictly positive orthant of packed length `n`.
    /// manopt `positivefactory`. Not the sphere and not a 3N cluster.
    Positive {
        /// Packed length. MATLAB `m*n` with default second size 1.
        n: usize,
    },
    /// Centered `m x n` matrices, packed row-major `m n`.
    /// manopt `centeredmatrixfactory`. Not the reserved SPD /
    /// grassmann / hyperbolic tokens.
    CenteredMatrix {
        /// Rows of the matrix. MATLAB `m`.
        m: usize,
        /// Columns of the matrix. MATLAB `n`.
        n: usize,
        /// `true` = centered rows. `false` = centered columns
        /// (manopt default).
        rows: bool,
    },
}

impl ManifoldKind {
    /// Stiefel \(\mathrm{St}(n,p)\). `p = 1` is the sphere packing.
    pub fn stiefel(n: usize, p: usize) -> Self {
        if p <= 1 {
            Self::Stiefel
        } else {
            Self::StiefelP { n, p }
        }
    }

    /// Stiefel column count. `p = 1` is the sphere.
    pub fn stiefel_p(self) -> usize {
        match self {
            Self::StiefelP { p, .. } => p,
            _ => 1,
        }
    }

    /// Oblique \(\mathrm{OB}(n, m)\): product of `m` unit spheres.
    pub fn oblique(n: usize, m: usize) -> Self {
        Self::Oblique { n, m }
    }

    /// Product of `n` unit circles. Packed length `2 n`.
    pub fn complex_circle(n: usize) -> Self {
        Self::ComplexCircle { n }
    }

    /// Complex Euclidean \(\mathbb{C}^n\). Packed length `2 n`.
    pub fn euclidean_complex(n: usize) -> Self {
        Self::EuclideanComplex { n }
    }

    /// Singleton of packed length `n`. manopt `constantfactory`.
    pub fn constant(n: usize) -> Self {
        Self::Constant { n }
    }

    /// Doubly-stochastic n-by-n, packed length `n^2`.
    /// manopt `multinomialdoublystochasticfactory`.
    pub fn multinomial_ds(n: usize) -> Self {
        Self::MultinomialDoublyStochastic { n }
    }

    /// Symmetric doubly-stochastic n-by-n, packed length `n^2`.
    /// manopt `multinomialsymmetricfactory`.
    pub fn multinomial_sym(n: usize) -> Self {
        Self::MultinomialSymmetric { n }
    }

    /// Complex unit sphere in \(\mathbb{C}^n\). Packed length `2 n`.
    /// manopt `spherecomplexfactory`.
    pub fn sphere_complex(n: usize) -> Self {
        Self::SphereComplex { n }
    }

    /// Positive orthant of packed length `n`. manopt `positivefactory`.
    pub fn positive(n: usize) -> Self {
        Self::Positive { n }
    }

    /// Centered `m x n` matrices. manopt `centeredmatrixfactory`.
    /// `rows = false` centers columns (`X 1 = 0`).
    pub fn centered_matrix(m: usize, n: usize, rows: bool) -> Self {
        Self::CenteredMatrix { m, n, rows }
    }

    /// [`Manifold::project`] written back into `v`. The sphere does it
    /// without allocating, with the arithmetic of `Sphere::project`.
    pub(crate) fn project_in_place(self, x: &Array1<f64>, v: &mut Array1<f64>) {
        if self == Self::Sphere {
            let s: f64 = x.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            v.zip_mut_with(x, |vi, &xi| *vi -= s * xi);
        } else {
            *v = self.project(x, v);
        }
    }

    /// `true` when the retraction keeps a straight step on the set, so a
    /// chord can be capped and clipped after it. A curved kind caps the
    /// tangent step before the retraction instead.
    pub(crate) fn is_flat(self) -> bool {
        match self {
            Self::Euclidean
            | Self::RigidQuotient
            | Self::MwRigid
            | Self::Symmetric
            | Self::SkewSymmetric
            | Self::EuclideanComplex { .. }
            | Self::CenteredMatrix { .. }
            | Self::Constant { .. } => true,
            Self::Sphere
            | Self::So3
            | Self::Stiefel
            | Self::Se3
            | Self::Oblique { .. }
            | Self::Multinomial
            | Self::StiefelP { .. }
            | Self::Spd
            | Self::ComplexCircle { .. }
            | Self::MultinomialDoublyStochastic { .. }
            | Self::MultinomialSymmetric { .. }
            | Self::SphereComplex { .. }
            | Self::Positive { .. } => false,
        }
    }

    /// C ABI / INI token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Sphere => "sphere",
            Self::So3 => "so3",
            Self::Stiefel | Self::StiefelP { .. } => "stiefel",
            Self::Se3 => "se3",
            Self::RigidQuotient => "rigid_quotient",
            Self::MwRigid => "mw_rigid",
            Self::Oblique { .. } => "oblique",
            Self::Multinomial => "multinomial",
            Self::Spd => "spd",
            Self::Symmetric => "symmetric",
            Self::SkewSymmetric => "skewsymmetric",
            Self::ComplexCircle { .. } => "complex_circle",
            Self::EuclideanComplex { .. } => "euclidean_complex",
            Self::Constant { .. } => "constant",
            Self::MultinomialDoublyStochastic { .. } => "multinomialdoublystochastic",
            Self::MultinomialSymmetric { .. } => "multinomialsymmetric",
            Self::SphereComplex { .. } => "spherecomplex",
            Self::Positive { .. } => "positive",
            Self::CenteredMatrix { .. } => "centeredmatrix",
        }
    }
}

/// manopt_cpp `AbstractManifold` on a rank-1 f64 vector.
pub trait Manifold {
    /// `Ok` if `n` is a legal packing for this geometry.
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        let _ = n;
        Ok(())
    }
    /// Tangent projection of an ambient vector at `x`. Same length as `v`.
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    /// Ambient Euclidean gradient to a Riemannian gradient at `x`.
    /// Embedded manifolds with a Euclidean ambient metric use [`Self::project`].
    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        self.project(x, egrad)
    }
    /// Retraction of the tangent step `v` at `x`. Same length as `x`.
    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    /// Vector transport of `v` from `x_from` to `x_to`.
    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
}

impl Manifold for ManifoldKind {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        with_kind!(self, |f| f.required_dim(n))
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.project(x, v))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.retract(x, v))
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.transport(x_from, x_to, v))
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        with_kind!(self, |f| f.egrad2rgrad(x, egrad))
    }
}
