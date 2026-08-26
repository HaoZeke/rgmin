//! Closed ELPA stage and block-size tokens. These compile without
//! a linked ELPA. Kind 7 is 1-stage, kind 8 is 2-stage. GPU is a
//! runtime probe of the linked build, not a third kind.

use crate::lowest_mode::EigensolverKind;

/// ELPA solver stage. Integers match `ELPA_SOLVER_1STAGE` / `ELPA_SOLVER_2STAGE`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElpaStage {
    /// `ELPA_SOLVER_1STAGE`.
    OneStage = 1,
    /// `ELPA_SOLVER_2STAGE`.
    TwoStage = 2,
}

impl ElpaStage {
    /// Schema / C ABI name. Never an `elpa_set` key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::OneStage => "oneStage",
            Self::TwoStage => "twoStage",
        }
    }

    /// Decode a closed ELPA solver integer. Unknown values are `None`.
    pub const fn from_solver(raw: i32) -> Option<Self> {
        match raw {
            1 => Some(Self::OneStage),
            2 => Some(Self::TwoStage),
            _ => None,
        }
    }

    /// Stage for [`EigensolverKind::Elpa`] / [`EigensolverKind::Elpa2`].
    pub const fn from_kind(kind: EigensolverKind) -> Option<Self> {
        match kind {
            EigensolverKind::Elpa => Some(Self::OneStage),
            EigensolverKind::Elpa2 => Some(Self::TwoStage),
            _ => None,
        }
    }

    /// Integer written to the linked ELPA handle. Not a string.
    pub const fn elpa_solver(self) -> i32 {
        self as i32
    }
}

/// Typed ELPA extras. `nblk == 0` selects 16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElpaParams {
    /// ScaLAPACK block size. 0 selects 16.
    pub nblk: u32,
}

impl Default for ElpaParams {
    fn default() -> Self {
        Self { nblk: 0 }
    }
}

impl ElpaParams {
    /// Block size after the 0 -> 16 default.
    pub const fn nblk_or_default(self) -> u32 {
        if self.nblk == 0 {
            16
        } else {
            self.nblk
        }
    }
}

/// Closed ELPA integers for a dense call: `(solver, nblk)`.
pub fn elpa_config(kind: EigensolverKind, elpa: &ElpaParams) -> Option<(i32, u32)> {
    let stage = ElpaStage::from_kind(kind)?;
    Some((stage.elpa_solver(), elpa.nblk_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elpa_kind_maps_to_solver_integers() {
        assert_eq!(ElpaStage::from_kind(EigensolverKind::Elpa), Some(ElpaStage::OneStage));
        assert_eq!(ElpaStage::from_kind(EigensolverKind::Elpa2), Some(ElpaStage::TwoStage));
        assert_eq!(ElpaStage::OneStage.elpa_solver(), 1);
        assert_eq!(ElpaStage::TwoStage.elpa_solver(), 2);
        assert_eq!(ElpaStage::from_solver(1), Some(ElpaStage::OneStage));
        assert_eq!(ElpaStage::from_solver(2), Some(ElpaStage::TwoStage));
        assert!(ElpaStage::from_solver(0).is_none());
        assert!(ElpaStage::from_kind(EigensolverKind::Slate).is_none());
        assert!(ElpaStage::from_kind(EigensolverKind::Lanczos).is_none());
        assert_eq!(elpa_config(EigensolverKind::Elpa, &ElpaParams::default()), Some((1, 16)));
        assert_eq!(
            elpa_config(EigensolverKind::Elpa2, &ElpaParams { nblk: 32 }),
            Some((2, 32))
        );
        assert_eq!(ElpaParams::default().nblk_or_default(), 16);
        assert_eq!(EigensolverKind::from_ordinal(7), Some(EigensolverKind::Elpa));
        assert_eq!(EigensolverKind::from_ordinal(8), Some(EigensolverKind::Elpa2));
        assert!(EigensolverKind::from_ordinal(15).is_none());
        assert!(!EigensolverKind::Elpa.is_linked());
        assert!(!EigensolverKind::Elpa2.is_linked());
    }

    #[test]
    fn elpa_public_sources_have_no_string_keys() {
        let schema = include_str!("../schema/eigen.capnp");
        let schema_code = schema
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!schema_code.contains(": Text"), "schema must not declare a Text field");
        assert!(!schema_code.contains("elpa_set"));
        let impl_src = include_str!("elpa_kind.rs");
        let impl_only = impl_src.split("#[cfg(test)]").next().expect("impl");
        for src in [
            impl_only,
            include_str!("lowest_mode.rs"),
            include_str!("error.rs"),
            include_str!("ffi.rs"),
        ] {
            for line in src.lines() {
                let t = line.trim();
                if t.starts_with("//") || t.starts_with("///") {
                    continue;
                }
                assert!(!t.contains("elpa_set"), "elpa_set in {t}");
                assert!(!t.contains("ELPA_DEFAULT"), "ELPA_DEFAULT in {t}");
                assert!(!t.contains("Elpa2Cuda"), "Elpa2Cuda in {t}");
                assert!(!t.contains("Elpa2Hip"), "Elpa2Hip in {t}");
            }
        }
    }
}
