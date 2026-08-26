//! SLEPc EPS waist: MatShell plus typed [`EPSSet*`] / [`STSet*`] only.
//!
//! The PETSc options database is not a config channel. This module
//! never calls `EPSSetFromOptions`, `STSetFromOptions`, or
//! `PetscOptions*`. Unbuilt `slepc` stays [`Error::EigenUnavailable`].
//! A host that already lives in PETSc may pass a Pmat through
//! [`SlepcHost`].

use std::os::raw::c_void;

use ndarray::ArrayView1;

use crate::error::{Error, Result};
use crate::lowest_mode::{ApplyHessian, EigenParams, LowestMode};

/// SLEPc `EPSWhich`. Integers match `slepceps.h`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlepcWhich {
    /// `EPS_LARGEST_MAGNITUDE`
    LargestMagnitude = 1,
    /// `EPS_SMALLEST_MAGNITUDE`
    SmallestMagnitude = 2,
    /// `EPS_LARGEST_REAL`
    LargestReal = 3,
    /// `EPS_SMALLEST_REAL`
    SmallestReal = 4,
    /// `EPS_LARGEST_IMAGINARY`
    LargestImaginary = 5,
    /// `EPS_SMALLEST_IMAGINARY`
    SmallestImaginary = 6,
    /// `EPS_TARGET_MAGNITUDE`
    TargetMagnitude = 7,
    /// `EPS_TARGET_REAL`
    TargetReal = 8,
    /// `EPS_TARGET_IMAGINARY`
    TargetImaginary = 9,
    /// `EPS_ALL`
    All = 10,
    /// `EPS_WHICH_USER`
    WhichUser = 11,
}

/// SLEPc `EPSProblemType`. Integers match `slepceps.h`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlepcProblem {
    /// `EPS_HEP` (real-symmetric Hessian).
    Hep = 1,
    /// `EPS_GHEP`
    Ghep = 2,
    /// `EPS_NHEP`
    Nhep = 3,
    /// `EPS_GNHEP`
    Gnhep = 4,
    /// `EPS_PGNHEP`
    Pgnhep = 5,
    /// `EPS_GHIEP`
    Ghiep = 6,
    /// `EPS_BSE`
    Bse = 7,
    /// `EPS_HAMILT`
    Hamilt = 8,
    /// `EPS_LREP`
    Lrep = 9,
}

/// Closed ST kind. Official `STType` literals stay inside the crate.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlepcStKind {
    /// `STSHIFT`
    Shift = 0,
    /// `STSINVERT`
    Sinvert = 1,
    /// `STPRECOND`
    Precond = 2,
}

impl SlepcStKind {
    /// Official SLEPc `STType` for [`STSetType`]. Not a host string key.
    pub const fn as_sttype(self) -> *const u8 {
        match self {
            Self::Shift => b"shift\0".as_ptr(),
            Self::Sinvert => b"sinvert\0".as_ptr(),
            Self::Precond => b"precond\0".as_ptr(),
        }
    }
}

/// Host PETSc handles for [`crate::EigensolverKind::Slepc`].
#[derive(Clone, Copy, Debug)]
pub struct SlepcHost {
    /// Host PETSc `Mat` used as the ST Pmat. Null if the host has none.
    pub pmat: *mut c_void,
}

impl Default for SlepcHost {
    fn default() -> Self {
        Self {
            pmat: std::ptr::null_mut(),
        }
    }
}

/// Typed EPSSet* / STSet* plan. No options-database keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlepcPlan {
    /// [`EPSSetWhichEigenpairs`].
    pub which: SlepcWhich,
    /// [`EPSSetProblemType`].
    pub problem: SlepcProblem,
    /// [`EPSSetDimensions`] `nev`.
    pub nev: i32,
    /// [`EPSSetDimensions`] `ncv`.
    pub ncv: i32,
    /// [`EPSSetTolerances`] `max_it`.
    pub max_it: i32,
    /// [`EPSSetTolerances`] residual tolerance.
    pub tol: f64,
    /// [`STSetType`] via [`SlepcStKind::as_sttype`].
    pub st: SlepcStKind,
    /// Whether [`STSetPreconditionerMat`] runs.
    pub has_pmat: bool,
}

impl SlepcPlan {
    /// IRC kick: Hermitian, smallest real, MatShell A. Pmat selects STPRECOND.
    pub fn from_params(params: &EigenParams, n: usize, has_pmat: bool) -> Self {
        let nev = params.nev.max(1).min(n.max(1)) as i32;
        let ncv = (params.krylov_dim(n) as i32).clamp(nev + 1, n.max(1) as i32);
        Self {
            which: SlepcWhich::SmallestReal,
            problem: SlepcProblem::Hep,
            nev,
            ncv,
            max_it: params.iterations(n) as i32,
            tol: params.tolerance(),
            st: if has_pmat {
                SlepcStKind::Precond
            } else {
                SlepcStKind::Shift
            },
            has_pmat,
        }
    }
}

/// Dispatch for [`crate::lowest_mode`] with an optional host Pmat.
pub fn lowest_mode_slepc<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    host: SlepcHost,
) -> Result<LowestMode> {
    lowest(h, x, seed, params, host)
}

pub(crate) fn lowest<H: ApplyHessian + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    params: &EigenParams,
    host: SlepcHost,
) -> Result<LowestMode> {
    let _ = SlepcPlan::from_params(params, seed.len(), !host.pmat.is_null());
    #[cfg(not(feature = "slepc"))]
    {
        let _ = (h, x, seed, params, host);
        Err(Error::EigenUnavailable { kind: "slepc" })
    }
    #[cfg(feature = "slepc")]
    {
        apply::lowest(h, x, seed, params, host)
    }
}

#[cfg(feature = "slepc")]
mod apply {
    use super::*;
    use std::cell::Cell;
    use std::ptr;

    use ndarray::Array1;

    type PetscErrorCode = i32;
    type PetscInt = i32;
    type PetscReal = f64;
    type PetscScalar = f64;
    type PetscBool = i32;
    type Mat = *mut c_void;
    type Vec = *mut c_void;
    type Eps = *mut c_void;
    type St = *mut c_void;

    const MATOP_MULT: i32 = 3;
    const PETSC_DECIDE: PetscInt = -1;

    struct ShellCtx<'a> {
        apply: Box<dyn Fn(&[f64]) -> Array1<f64> + 'a>,
        n: usize,
        actions: Cell<usize>,
    }

    unsafe extern "C" {
        fn rgmin_slepc_initialized(flag: *mut PetscBool) -> PetscErrorCode;
        fn rgmin_slepc_eps_create(eps: *mut Eps) -> PetscErrorCode;
        fn rgmin_slepc_mat_create_shell(n: i64, ctx: *mut c_void, a: *mut Mat) -> PetscErrorCode;
        fn MatShellSetOperation(mat: Mat, op: i32, f: *const c_void) -> PetscErrorCode;
        fn MatShellGetContext(mat: Mat, ctx: *mut *mut c_void) -> PetscErrorCode;
        fn MatCreateVecs(mat: Mat, right: *mut Vec, left: *mut Vec) -> PetscErrorCode;
        fn MatDestroy(mat: *mut Mat) -> PetscErrorCode;
        fn VecGetArrayRead(x: Vec, a: *mut *const PetscScalar) -> PetscErrorCode;
        fn VecRestoreArrayRead(x: Vec, a: *mut *const PetscScalar) -> PetscErrorCode;
        fn VecGetArray(x: Vec, a: *mut *mut PetscScalar) -> PetscErrorCode;
        fn VecRestoreArray(x: Vec, a: *mut *mut PetscScalar) -> PetscErrorCode;
        fn VecDestroy(x: *mut Vec) -> PetscErrorCode;
        fn EPSSetOperators(eps: Eps, a: Mat, b: Mat) -> PetscErrorCode;
        fn EPSSetProblemType(eps: Eps, t: i32) -> PetscErrorCode;
        fn EPSSetWhichEigenpairs(eps: Eps, w: i32) -> PetscErrorCode;
        fn EPSSetDimensions(
            eps: Eps,
            nev: PetscInt,
            ncv: PetscInt,
            mpd: PetscInt,
        ) -> PetscErrorCode;
        fn EPSSetTolerances(eps: Eps, tol: PetscReal, max_it: PetscInt) -> PetscErrorCode;
        fn EPSSetInitialSpace(eps: Eps, n: PetscInt, is: *mut Vec) -> PetscErrorCode;
        fn EPSGetST(eps: Eps, st: *mut St) -> PetscErrorCode;
        fn STSetType(st: St, t: *const u8) -> PetscErrorCode;
        fn STSetPreconditionerMat(st: St, mat: Mat) -> PetscErrorCode;
        fn EPSSolve(eps: Eps) -> PetscErrorCode;
        fn EPSGetConverged(eps: Eps, nconv: *mut PetscInt) -> PetscErrorCode;
        fn EPSGetEigenpair(
            eps: Eps,
            i: PetscInt,
            kr: *mut PetscScalar,
            ki: *mut PetscScalar,
            xr: Vec,
            xi: Vec,
        ) -> PetscErrorCode;
        fn EPSDestroy(eps: *mut Eps) -> PetscErrorCode;
    }

    fn chk(err: PetscErrorCode, what: &'static str) -> Result<()> {
        if err == 0 {
            Ok(())
        } else {
            Err(Error::Oracle { what })
        }
    }

    unsafe extern "C" fn mat_mult(a: Mat, x: Vec, y: Vec) -> PetscErrorCode {
        let mut raw = ptr::null_mut();
        if MatShellGetContext(a, &mut raw) != 0 || raw.is_null() {
            return 1;
        }
        let ctx = unsafe { &*(raw as *const ShellCtx<'_>) };
        let mut xin = ptr::null();
        let mut yout = ptr::null_mut();
        if VecGetArrayRead(x, &mut xin) != 0 {
            return 1;
        }
        if VecGetArray(y, &mut yout) != 0 {
            let _ = VecRestoreArrayRead(x, &mut xin);
            return 1;
        }
        let v = unsafe { std::slice::from_raw_parts(xin, ctx.n) };
        let hv = (ctx.apply)(v);
        ctx.actions.set(ctx.actions.get() + 1);
        let out = unsafe { std::slice::from_raw_parts_mut(yout, ctx.n) };
        if let Some(src) = hv.as_slice() {
            out.copy_from_slice(src);
        }
        let mut xin_r = xin;
        let mut yout_r = yout;
        let e1 = VecRestoreArrayRead(x, &mut xin_r);
        let e2 = VecRestoreArray(y, &mut yout_r);
        if e1 != 0 || e2 != 0 { 1 } else { 0 }
    }

    pub(super) fn lowest<H: ApplyHessian + ?Sized>(
        h: &H,
        x: ArrayView1<f64>,
        seed: ArrayView1<f64>,
        params: &EigenParams,
        host: SlepcHost,
    ) -> Result<LowestMode> {
        let n = seed.len();
        if n == 0 {
            return Err(Error::Dim { got: 0, dim: 0 });
        }
        let mut inited: PetscBool = 0;
        chk(
            unsafe { rgmin_slepc_initialized(&mut inited) },
            "slepc PetscInitialized",
        )?;
        if inited == 0 {
            return Err(Error::EigenUnavailable { kind: "slepc" });
        }
        let plan = SlepcPlan::from_params(params, n, !host.pmat.is_null());
        let x0 = x.to_owned();
        let mut ctx = ShellCtx {
            apply: Box::new(move |v: &[f64]| h.apply_hessian(x0.view(), ArrayView1::from(v))),
            n,
            actions: Cell::new(0),
        };
        let mut a: Mat = ptr::null_mut();
        chk(
            unsafe { rgmin_slepc_mat_create_shell(n as i64, (&raw mut ctx).cast(), &mut a) },
            "slepc MatCreateShell",
        )?;
        let mult: unsafe extern "C" fn(Mat, Vec, Vec) -> PetscErrorCode = mat_mult;
        let set_op = unsafe { MatShellSetOperation(a, MATOP_MULT, mult as *const c_void) };
        if set_op != 0 {
            unsafe {
                let _ = MatDestroy(&mut a);
            }
            return Err(Error::Oracle {
                what: "slepc MatShellSetOperation",
            });
        }
        let mut eps: Eps = ptr::null_mut();
        let created = unsafe { rgmin_slepc_eps_create(&mut eps) };
        if created != 0 {
            unsafe {
                let _ = MatDestroy(&mut a);
            }
            return Err(Error::Oracle {
                what: "slepc EPSCreate",
            });
        }
        let run = (|| -> Result<LowestMode> {
            chk(
                unsafe { EPSSetOperators(eps, a, ptr::null_mut()) },
                "slepc EPSSetOperators",
            )?;
            chk(
                unsafe { EPSSetProblemType(eps, plan.problem as i32) },
                "slepc EPSSetProblemType",
            )?;
            chk(
                unsafe { EPSSetWhichEigenpairs(eps, plan.which as i32) },
                "slepc EPSSetWhichEigenpairs",
            )?;
            chk(
                unsafe { EPSSetDimensions(eps, plan.nev, plan.ncv, PETSC_DECIDE) },
                "slepc EPSSetDimensions",
            )?;
            chk(
                unsafe { EPSSetTolerances(eps, plan.tol, plan.max_it) },
                "slepc EPSSetTolerances",
            )?;
            let mut st: St = ptr::null_mut();
            chk(unsafe { EPSGetST(eps, &mut st) }, "slepc EPSGetST")?;
            chk(
                unsafe { STSetType(st, plan.st.as_sttype()) },
                "slepc STSetType",
            )?;
            if plan.has_pmat {
                chk(
                    unsafe { STSetPreconditionerMat(st, host.pmat) },
                    "slepc STSetPreconditionerMat",
                )?;
            }
            let mut v0: Vec = ptr::null_mut();
            chk(
                unsafe { MatCreateVecs(a, &mut v0, ptr::null_mut()) },
                "slepc MatCreateVecs",
            )?;
            let mut v0a = ptr::null_mut();
            let seed_copy = (|| -> Result<()> {
                chk(unsafe { VecGetArray(v0, &mut v0a) }, "slepc VecGetArray")?;
                let buf = unsafe { std::slice::from_raw_parts_mut(v0a, n) };
                buf.copy_from_slice(seed.as_slice().unwrap_or(&[]));
                let mut tmp = v0a;
                chk(
                    unsafe { VecRestoreArray(v0, &mut tmp) },
                    "slepc VecRestoreArray",
                )
            })();
            if let Err(e) = seed_copy {
                unsafe {
                    let _ = VecDestroy(&mut v0);
                }
                return Err(e);
            }
            let set_is = unsafe { EPSSetInitialSpace(eps, 1, &mut v0) };
            unsafe {
                let _ = VecDestroy(&mut v0);
            }
            chk(set_is, "slepc EPSSetInitialSpace")?;
            chk(unsafe { EPSSolve(eps) }, "slepc EPSSolve")?;
            let mut nconv: PetscInt = 0;
            chk(
                unsafe { EPSGetConverged(eps, &mut nconv) },
                "slepc EPSGetConverged",
            )?;
            if nconv < 1 {
                return Err(Error::Oracle {
                    what: "slepc no converged pair",
                });
            }
            let mut xr: Vec = ptr::null_mut();
            chk(
                unsafe { MatCreateVecs(a, &mut xr, ptr::null_mut()) },
                "slepc MatCreateVecs xr",
            )?;
            let mut kr: PetscScalar = 0.0;
            let mut ki: PetscScalar = 0.0;
            let got = unsafe { EPSGetEigenpair(eps, 0, &mut kr, &mut ki, xr, ptr::null_mut()) };
            if got != 0 {
                unsafe {
                    let _ = VecDestroy(&mut xr);
                }
                return Err(Error::Oracle {
                    what: "slepc EPSGetEigenpair",
                });
            }
            let mut xa = ptr::null();
            let mode = (|| -> Result<Array1<f64>> {
                chk(
                    unsafe { VecGetArrayRead(xr, &mut xa) },
                    "slepc VecGetArrayRead",
                )?;
                let buf = unsafe { std::slice::from_raw_parts(xa, n) };
                let v = Array1::from(buf.to_vec());
                let mut tmp = xa;
                chk(
                    unsafe { VecRestoreArrayRead(xr, &mut tmp) },
                    "slepc VecRestoreArrayRead",
                )?;
                Ok(v)
            })();
            unsafe {
                let _ = VecDestroy(&mut xr);
            }
            let vector = mode?;
            let nrm = crate::vecops::nrm2(vector.view());
            let vector = if nrm > 0.0 && nrm.is_finite() {
                vector / nrm
            } else {
                vector
            };
            Ok(LowestMode {
                vector,
                value: kr,
                actions: ctx.actions.get(),
            })
        })();
        unsafe {
            let _ = EPSDestroy(&mut eps);
            let _ = MatDestroy(&mut a);
        }
        run
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowest_mode::EigensolverKind;
    use ndarray::{array, Array1};

    struct DiagH(Array1<f64>);
    impl ApplyHessian for DiagH {
        fn apply_hessian(&self, _x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter(self.0.iter().zip(v.iter()).map(|(l, vi)| l * vi))
        }
    }

    #[test]
    fn plan_uses_typed_smallest_real_hep() {
        let params = EigenParams {
            kind: EigensolverKind::Slepc,
            nev: 1,
            krylov: 8,
            max_iter: 20,
            tol: 1e-8,
        };
        let plan = SlepcPlan::from_params(&params, 16, false);
        assert_eq!(plan.which, SlepcWhich::SmallestReal);
        assert_eq!(plan.which as i32, 4);
        assert_eq!(plan.problem, SlepcProblem::Hep);
        assert_eq!(plan.problem as i32, 1);
        assert_eq!(plan.nev, 1);
        assert_eq!(plan.ncv, 8);
        assert_eq!(plan.max_it, 20);
        assert!((plan.tol - 1e-8).abs() < 1e-20);
        assert_eq!(plan.st, SlepcStKind::Shift);
        assert!(!plan.has_pmat);
    }

    #[test]
    fn plan_host_pmat_selects_stprecond() {
        let params = EigenParams {
            kind: EigensolverKind::Slepc,
            ..EigenParams::default()
        };
        let plan = SlepcPlan::from_params(&params, 8, true);
        assert_eq!(plan.st, SlepcStKind::Precond);
        assert!(plan.has_pmat);
        assert_eq!(plan.which, SlepcWhich::SmallestReal);
        assert_eq!(plan.problem, SlepcProblem::Hep);
    }

    #[test]
    fn sttype_literals_are_the_official_closed_map() {
        let shift =
            unsafe { std::ffi::CStr::from_ptr(SlepcStKind::Shift.as_sttype() as *const i8) };
        let sinvert =
            unsafe { std::ffi::CStr::from_ptr(SlepcStKind::Sinvert.as_sttype() as *const i8) };
        let precond =
            unsafe { std::ffi::CStr::from_ptr(SlepcStKind::Precond.as_sttype() as *const i8) };
        assert_eq!(shift.to_bytes(), b"shift");
        assert_eq!(sinvert.to_bytes(), b"sinvert");
        assert_eq!(precond.to_bytes(), b"precond");
    }

    #[test]
    fn waist_source_has_no_options_database() {
        let rust = include_str!("slepc_eps.rs");
        let shim = include_str!("slepc_shim.c");
        let eps_from = concat!("EPSSet", "FromOptions(");
        let st_from = concat!("STSet", "FromOptions(");
        let opt_set = concat!("Petsc", "OptionsSet");
        let opt_ins = concat!("Petsc", "OptionsInsert");
        let petsc_init = concat!("Petsc", "Initialize(");
        for src in [rust, shim] {
            assert!(!src.contains(eps_from), "options-database EPS setter");
            assert!(!src.contains(st_from), "options-database ST setter");
            assert!(!src.contains(opt_set), "Petsc options set");
            assert!(!src.contains(opt_ins), "Petsc options insert");
            assert!(!src.contains(petsc_init), "PetscInitialize");
        }
        assert!(rust.contains("EPSSetProblemType"));
        assert!(rust.contains("EPSSetWhichEigenpairs"));
        assert!(rust.contains("EPSSetDimensions"));
        assert!(rust.contains("EPSSetTolerances"));
        assert!(rust.contains("STSetType"));
        assert!(rust.contains("STSetPreconditionerMat"));
        assert!(rust.contains("MatCreateShell") || shim.contains("MatCreateShell"));
    }

    #[cfg(not(feature = "slepc"))]
    #[test]
    fn unbuilt_feature_is_eigen_unavailable() {
        assert!(!EigensolverKind::Slepc.is_linked());
        let h = DiagH(array![1.0, 2.0, 3.0]);
        let x = Array1::zeros(3);
        let seed = array![1.0, 0.0, 0.0];
        let err = lowest_mode_slepc(
            &h,
            x.view(),
            seed.view(),
            &EigenParams {
                kind: EigensolverKind::Slepc,
                ..EigenParams::default()
            },
            SlepcHost::default(),
        )
        .unwrap_err();
        match err {
            Error::EigenUnavailable { kind } => assert_eq!(kind, "slepc"),
            other => panic!("expected unavailable, got {other}"),
        }
    }
}
