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
//! / Symmetric-n² / SPD-n² / ComplexCircle-2n / Unitary-2n² /
//! Oblique-nm are matrix-manifold embeddings, not a 3N cluster.
//! [`ManifoldKind::Symmetric`] is manopt `symmetricfactory`.
//! [`ManifoldKind::ComplexCircle`] is manopt `complexcirclefactory`.
//! [`ManifoldKind::Unitary`] is manopt `unitaryfactory`.

use ndarray::Array1;

mod complex_circle;
mod euclidean;
mod multinomial;
mod mw_rigid;
mod oblique;
mod rigid_quotient;
mod se3;
mod so3;
mod spd;
mod sphere;
mod stiefel;
mod symmetric;
mod unitary;

pub use complex_circle::ComplexCircle;
pub use euclidean::Euclidean;
pub use multinomial::Multinomial;
pub use mw_rigid::MwRigid;
pub use oblique::Oblique;
pub use rigid_quotient::RigidQuotient;
pub use se3::Se3;
pub use so3::So3;
pub use spd::{is_spd, pack as pack_spd, side as side_spd, unpack as unpack_spd, Spd};
pub use sphere::Sphere;
pub use stiefel::{Stiefel, StiefelNp};
pub use symmetric::{
    inner as inner_sym, is_symmetric, pack as pack_sym, side as side_sym,
    typical_dist as typical_dist_sym, unpack as unpack_sym, Symmetric,
};
pub use unitary::{
    is_unitary, pack as pack_unitary, side as side_unitary, unpack as unpack_unitary, Unitary,
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
    /// Product of unit-modulus complex numbers \((S^1)^n\).
    /// Packed interleaved `(re, im)`, length `2 n`.
    /// manopt `complexcirclefactory(n)`. Not the sphere.
    ComplexCircle {
        /// Number of unit-modulus complex entries.
        n: usize,
    },
    /// Unitary group \(\mathrm{U}(n)\) for `n >= 1`. Packed
    /// interleaved `(re, im)` row-major, length `2 n^2`.
    /// manopt `unitaryfactory(n)` at `k = 1`. Construct with
    /// [`ManifoldKind::unitary`].
    Unitary {
        /// Matrix side. Must be `>= 1`.
        n: usize,
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

    /// \(\mathrm{U}(n)\), packed interleaved row-major length `2 n^2`.
    pub fn unitary(n: usize) -> Self {
        Self::Unitary { n }
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
            Self::ComplexCircle { .. } => "complex_circle",
            Self::Unitary { .. } => "unitary",
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
        match self {
            Self::Euclidean => Euclidean.required_dim(n),
            Self::Sphere => Sphere.required_dim(n),
            Self::So3 => So3.required_dim(n),
            Self::Stiefel => Stiefel.required_dim(n),
            Self::Se3 => Se3.required_dim(n),
            Self::RigidQuotient => RigidQuotient.required_dim(n),
            Self::MwRigid => MwRigid.required_dim(n),
            Self::Oblique { n: an, m } => Oblique { n: *an, m: *m }.required_dim(n),
            Self::Multinomial => Multinomial.required_dim(n),
            Self::StiefelP { n: sn, p } => StiefelNp { n: *sn, p: *p }.required_dim(n),
            Self::Spd => Spd.required_dim(n),
            Self::Symmetric => Symmetric.required_dim(n),
            Self::ComplexCircle { n: cn } => ComplexCircle { n: *cn }.required_dim(n),
            Self::Unitary { n: un } => Unitary { n: *un }.required_dim(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.project(x, v),
            Self::Sphere => Sphere.project(x, v),
            Self::So3 => So3.project(x, v),
            Self::Stiefel => Stiefel.project(x, v),
            Self::Se3 => Se3.project(x, v),
            Self::RigidQuotient => RigidQuotient.project(x, v),
            Self::MwRigid => MwRigid.project(x, v),
            Self::Oblique { n, m } => Oblique { n: *n, m: *m }.project(x, v),
            Self::Multinomial => Multinomial.project(x, v),
            Self::StiefelP { n, p } => StiefelNp { n: *n, p: *p }.project(x, v),
            Self::Spd => Spd.project(x, v),
            Self::Symmetric => Symmetric.project(x, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.project(x, v),
            Self::Unitary { n } => Unitary { n: *n }.project(x, v),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.retract(x, v),
            Self::Sphere => Sphere.retract(x, v),
            Self::So3 => So3.retract(x, v),
            Self::Stiefel => Stiefel.retract(x, v),
            Self::Se3 => Se3.retract(x, v),
            Self::RigidQuotient => RigidQuotient.retract(x, v),
            Self::MwRigid => MwRigid.retract(x, v),
            Self::Oblique { n, m } => Oblique { n: *n, m: *m }.retract(x, v),
            Self::Multinomial => Multinomial.retract(x, v),
            Self::StiefelP { n, p } => StiefelNp { n: *n, p: *p }.retract(x, v),
            Self::Spd => Spd.retract(x, v),
            Self::Symmetric => Symmetric.retract(x, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.retract(x, v),
            Self::Unitary { n } => Unitary { n: *n }.retract(x, v),
        }
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.transport(x_from, x_to, v),
            Self::Sphere => Sphere.transport(x_from, x_to, v),
            Self::So3 => So3.transport(x_from, x_to, v),
            Self::Stiefel => Stiefel.transport(x_from, x_to, v),
            Self::Se3 => Se3.transport(x_from, x_to, v),
            Self::RigidQuotient => RigidQuotient.transport(x_from, x_to, v),
            Self::MwRigid => MwRigid.transport(x_from, x_to, v),
            Self::Oblique { n, m } => Oblique { n: *n, m: *m }.transport(x_from, x_to, v),
            Self::Multinomial => Multinomial.transport(x_from, x_to, v),
            Self::StiefelP { n, p } => StiefelNp { n: *n, p: *p }.transport(x_from, x_to, v),
            Self::Spd => Spd.transport(x_from, x_to, v),
            Self::Symmetric => Symmetric.transport(x_from, x_to, v),
            Self::ComplexCircle { n } => ComplexCircle { n: *n }.transport(x_from, x_to, v),
            Self::Unitary { n } => Unitary { n: *n }.transport(x_from, x_to, v),
        }
    }
}
