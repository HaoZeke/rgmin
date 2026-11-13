//! The C direction takes stored pairs and a caller-supplied H0.
//! A null handle is an error and does not panic.

#![cfg(feature = "capi")]

use rgmin::ffi::{
    rgmin_lbfgs_direction, rgmin_lbfgs_pairs_clear, rgmin_lbfgs_pairs_count,
    rgmin_lbfgs_pairs_create, rgmin_lbfgs_pairs_free, rgmin_lbfgs_pairs_push, rgmin_status_t,
};

unsafe extern "C" fn diag_h0(
    user: *mut std::ffi::c_void,
    v: *const f64,
    out: *mut f64,
    n: usize,
) -> i32 {
    let diag = unsafe { std::slice::from_raw_parts(user.cast::<f64>(), n) };
    let src = unsafe { std::slice::from_raw_parts(v, n) };
    let dst = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n {
        dst[i] = diag[i] * src[i];
    }
    0
}

unsafe extern "C" fn fail_h0(
    _user: *mut std::ffi::c_void,
    _v: *const f64,
    _out: *mut f64,
    _n: usize,
) -> i32 {
    1
}

#[test]
fn null_handles_and_bad_lengths_are_errors() {
    let mut dir = [0.0; 2];
    let grad = [1.0, -1.0];
    let diag = [0.5, 0.25];
    let status = unsafe {
        rgmin_lbfgs_direction(
            std::ptr::null(),
            grad.as_ptr(),
            diag.as_ptr(),
            None,
            std::ptr::null_mut(),
            dir.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    unsafe {
        rgmin_lbfgs_pairs_free(std::ptr::null_mut());
        rgmin_lbfgs_pairs_clear(std::ptr::null_mut());
    }
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(std::ptr::null()) }, 0);
    assert!(unsafe { rgmin_lbfgs_pairs_create(0, 4) }.is_null());

    let pairs = unsafe { rgmin_lbfgs_pairs_create(2, 4) };
    assert!(!pairs.is_null());
    let both = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            diag.as_ptr(),
            Some(diag_h0),
            std::ptr::null_mut(),
            dir.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(both, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    let neither = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            std::ptr::null(),
            None,
            std::ptr::null_mut(),
            dir.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(neither, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    let short = [1.0];
    let mut stored = 1;
    let bad =
        unsafe { rgmin_lbfgs_pairs_push(pairs, short.as_ptr(), grad.as_ptr(), 1, &mut stored) };
    assert_eq!(bad, rgmin_status_t::RGMIN_INVALID_PARAMETER);
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(pairs) }, 0);
    unsafe { rgmin_lbfgs_pairs_free(pairs) };
}

#[test]
fn empty_memory_and_a_callback_match_the_diagonal() {
    let pairs = unsafe { rgmin_lbfgs_pairs_create(2, 0) };
    assert!(!pairs.is_null());
    let grad = [1.0, -2.0];
    let diag = [0.5, 0.25];
    let mut from_diag = [0.0; 2];
    let status = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            diag.as_ptr(),
            None,
            std::ptr::null_mut(),
            from_diag.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(from_diag, [-0.5, 0.5]);

    let mut from_callback = [0.0; 2];
    let status = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            std::ptr::null(),
            Some(diag_h0),
            diag.as_ptr() as *mut std::ffi::c_void,
            from_callback.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(status, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(from_callback, from_diag);

    let failed = unsafe {
        rgmin_lbfgs_direction(
            pairs,
            grad.as_ptr(),
            std::ptr::null(),
            Some(fail_h0),
            std::ptr::null_mut(),
            from_callback.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(failed, rgmin_status_t::RGMIN_INVALID_PARAMETER);

    let s = [1.0, 0.0];
    let y = [2.0, 1.0];
    let mut stored = 0;
    let pushed = unsafe { rgmin_lbfgs_pairs_push(pairs, s.as_ptr(), y.as_ptr(), 2, &mut stored) };
    assert_eq!(pushed, rgmin_status_t::RGMIN_SUCCESS);
    assert_eq!(stored, 1);
    assert_eq!(unsafe { rgmin_lbfgs_pairs_count(pairs) }, 1);
    unsafe { rgmin_lbfgs_pairs_free(pairs) };
}
