//! Issue 15: an L-BFGS callback receives the accepted coordinates.
//!
//! `minimize_watched` reports the iteration and the value. A path over a
//! surface needs the positions, so the caller can project them (here the
//! gap `x[1] - x[0]`, standing in for a bond length) without the solver
//! knowing that map.

use ndarray::{Array1, ArrayView1, array};
use rgmin::Lbfgs;

fn bowl(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let f = x[0] * x[0] + 4.0 * x[1] * x[1];
    let g = array![2.0 * x[0], 8.0 * x[1]];
    (f, g)
}

fn gap(x: ArrayView1<f64>) -> f64 {
    x[1] - x[0]
}

#[test]
fn position_callback_records_the_projected_path_and_stops_on_it() {
    let start = array![1.5, -0.8];
    let mut opt = Lbfgs::default();
    let mut seen: Vec<(f64, Array1<f64>, f64)> = Vec::new();
    let (f, x, _) = opt.minimize_on(
        start.view(),
        40,
        |v| Some(bowl(v)),
        |it, fv, xv| {
            seen.push((fv, xv.to_owned(), gap(xv)));
            it < 2
        },
    );
    assert!(seen.len() >= 2, "callback saw only {seen:?}");
    assert_eq!(seen[0].1, start);
    assert_eq!(seen[0].2, gap(start.view()));
    let (fv, xv, bond) = seen.last().unwrap();
    assert_eq!(f, *fv);
    assert_eq!(&x, xv);
    assert_eq!(*bond, gap(x.view()));
    assert_ne!(xv, &start, "later iterates stayed at the start");
    assert_ne!(seen[0].2, *bond, "the projected coordinate never moved");
}

#[test]
fn position_callback_that_accepts_every_point_matches_minimize() {
    let start = array![1.5, -0.8];
    let mut plain = Lbfgs::default();
    let (fa, xa, ea) = plain.minimize(start.view(), 40, |v| Some(bowl(v)));
    let mut traced = Lbfgs::default();
    let mut bonds = Vec::new();
    let (fb, xb, eb) = traced.minimize_on(
        start.view(),
        40,
        |v| Some(bowl(v)),
        |_, _, p| {
            bonds.push(gap(p));
            true
        },
    );
    assert_eq!((fa, ea), (fb, eb));
    assert_eq!(xa, xb);
    assert_eq!(bonds[0], gap(start.view()));
    assert!((bonds.last().copied().unwrap() - gap(xb.view())).abs() < 1e-12);
    assert!(fb < 1e-8, "did not reach the bowl, f = {fb}");
}

#[test]
fn box_position_callback_sees_the_clipped_start() {
    let start = array![-1.0, 2.0];
    let lower = array![0.0, -0.5];
    let upper = array![0.4, 3.0];
    let mut opt = Lbfgs::default();
    let mut seen = Vec::new();
    let (f, x, evals) = opt.minimize_box_on(
        start.view(),
        lower.view(),
        upper.view(),
        20,
        |v| Some(bowl(v)),
        |_it, fv, xv| {
            seen.push((fv, xv.to_owned()));
            false
        },
    );
    assert_eq!(seen.len(), 1);
    let (fv, xv) = &seen[0];
    assert_eq!(xv[0], 0.0);
    assert_eq!(xv[1], 2.0);
    assert_eq!(f, *fv);
    assert_eq!(&x, xv);
    assert_eq!(evals, 1);
}
