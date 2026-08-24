//! Riemannian trust region with truncated CG (manopt `trustregions` + `tCG`).
//!
//! Absil, Baker, Gallivan, *Trust-region methods on Riemannian manifolds*,
//! <https://doi.org/10.1007/s10208-005-0179-9>.
//! Boumal, Mishra, Absil, Sepulchre, *Manopt*,
//! <https://doi.org/10.5555/2627435.2638581>.
//! The inner solve is Steihaug--Toint CG in the tangent: same algebra as
//! [`crate::steihaug_cg`], with a caller-supplied Hessian action so the
//! Euclidean dense path and a finite-difference Riemannian Hessian share
//! one loop. The outer step retracts the tangent increment; radius and
//! ratio tests are [`crate::trust`].

use eindir_core::DifferentiableObjective;
use ndarray::Array1;

use crate::manifold::Manifold;
use crate::vecops::{axpy, dot, nrm2};

/// manopt `tCG` residual: `||r|| <= ||r0|| min(||r0||^theta, kappa)`.
pub const TCG_KAPPA: f64 = 0.1;
/// Forcing exponent in the manopt residual test.
pub const TCG_THETA: f64 = 1.0;

/// Truncated CG on `min_p g.p + p.Hp/2` subject to `||p|| <= radius`.
///
/// `hess_vec` is the (Riemannian) Hessian action at the current point,
/// already mapped into the tangent. Returns the step and the model
/// reduction `m(0) - m(p)`.
pub fn tcg<H>(grad: &Array1<f64>, radius: f64, maxiter: usize, hess_vec: H) -> (Array1<f64>, f64)
where
    H: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n = grad.len();
    let mut p = Array1::<f64>::zeros(n);
    let mut hp = Array1::<f64>::zeros(n);
    let mut r = grad.clone();
    let mut d = r.mapv(|v| -v);
    let mut rz = dot(r.view(), r.view());
    let gnorm = nrm2(grad.view());
    let stop = (gnorm * gnorm.powf(TCG_THETA).min(TCG_KAPPA)).max(f64::MIN_POSITIVE);

    let mut p_p = 0.0_f64;
    let mut p_d = 0.0_f64;
    let mut d_d = rz;
    let r2 = radius * radius;

    let model_drop = |p: &Array1<f64>, hp: &Array1<f64>| {
        -(dot(grad.view(), p.view()) + 0.5 * dot(p.view(), hp.view()))
    };
    let boundary_tau = |p_p: f64, p_d: f64, d_d: f64| -> f64 {
        if d_d <= 0.0 {
            return 0.0;
        }
        let disc = (p_d * p_d + d_d * (r2 - p_p)).max(0.0);
        (-p_d + disc.sqrt()) / d_d
    };

    if !(rz.is_finite() && rz > 0.0) {
        let p = grad.mapv(|v| -radius * v / gnorm.max(f64::MIN_POSITIVE));
        let hp = hess_vec(&p);
        let drop = model_drop(&p, &hp);
        return (p, drop);
    }

    for _ in 0..maxiter {
        let hd = hess_vec(&d);
        let dhd = dot(d.view(), hd.view());
        if !dhd.is_finite() {
            let p = grad.mapv(|v| -radius * v / gnorm.max(f64::MIN_POSITIVE));
            let hp = hess_vec(&p);
            let drop = model_drop(&p, &hp);
            return (p, drop);
        }
        if dhd <= 0.0 {
            let tau = boundary_tau(p_p, p_d, d_d);
            axpy(tau, d.view(), &mut p);
            axpy(tau, hd.view(), &mut hp);
            let drop = model_drop(&p, &hp);
            return (p, drop);
        }
        let alpha = rz / dhd;
        let p_p_next = p_p + 2.0 * alpha * p_d + alpha * alpha * d_d;
        if p_p_next >= r2 {
            let tau = boundary_tau(p_p, p_d, d_d);
            axpy(tau, d.view(), &mut p);
            axpy(tau, hd.view(), &mut hp);
            let drop = model_drop(&p, &hp);
            return (p, drop);
        }
        axpy(alpha, d.view(), &mut p);
        axpy(alpha, hd.view(), &mut hp);
        p_p = p_p_next;
        axpy(alpha, hd.view(), &mut r);
        if nrm2(r.view()) < stop {
            break;
        }
        let rz_next = dot(r.view(), r.view());
        if !(rz_next.is_finite() && rz_next > 0.0) {
            break;
        }
        let beta = rz_next / rz;
        p_d = beta * (p_d + alpha * d_d);
        d_d = rz_next + beta * beta * d_d;
        rz = rz_next;
        for (di, ri) in d.iter_mut().zip(r.iter()) {
            *di = -ri + beta * *di;
        }
    }
    let drop = model_drop(&p, &hp);
    (p, drop)
}

/// Finite-difference Riemannian Hessian action at `x` along tangent `eta`.
///
/// manopt `getHessianFD`: retract a scaled `eta`, pull the new Riemannian
/// gradient back by transport, divide. The result is projected at `x`.
pub fn rhess_fd<O, M>(
    obj: &O,
    man: &M,
    x: &Array1<f64>,
    rgrad: &Array1<f64>,
    eta: &Array1<f64>,
    eps: f64,
) -> Array1<f64>
where
    O: DifferentiableObjective<f64> + ?Sized,
    M: Manifold + ?Sized,
{
    let vn = nrm2(eta.view());
    if vn <= 0.0 {
        return Array1::zeros(eta.len());
    }
    let h = eps.max(1e-16) / vn;
    let mut step = Array1::zeros(eta.len());
    axpy(h, eta.view(), &mut step);
    let y = man.retract(x, &step);
    let gy = obj.value_and_gradient(y.view()).1;
    let rgy = man.egrad2rgrad(&y, &gy);
    let pulled = man.transport(&y, x, &rgy);
    let mut diff = pulled;
    axpy(-1.0, rgrad.view(), &mut diff);
    let scale = 1.0 / h;
    for v in diff.iter_mut() {
        *v *= scale;
    }
    man.project(x, &diff)
}

/// Dense Euclidean Hessian action, then the manifold's `ehess2rhess`.
pub fn rhess_ehess<M>(
    man: &M,
    x: &Array1<f64>,
    egrad: &Array1<f64>,
    hess: &ndarray::Array2<f64>,
    eta: &Array1<f64>,
) -> Array1<f64>
where
    M: Manifold + ?Sized,
{
    let ehess = hess.dot(eta);
    man.ehess2rhess(x, egrad, &ehess, eta)
}

/// Predicted reduction from a Hessian-vector product (no dense `H`).
pub fn predicted_reduction_hvp(grad: &Array1<f64>, p: &Array1<f64>, hp: &Array1<f64>) -> f64 {
    -dot(grad.view(), p.view()) - 0.5 * dot(p.view(), hp.view())
}

/// Scale a tangent increment so its Euclidean length is at most `cap`.
pub fn scale_tangent(eta: &mut Array1<f64>, cap: f64) {
    let n = nrm2(eta.view());
    if n > cap && n > 0.0 && cap > 0.0 {
        let s = cap / n;
        for v in eta.iter_mut() {
            *v *= s;
        }
    }
}

/// One FD action of a linear gradient is the exact Hessian action.
pub fn fd_eps() -> f64 {
    1e-7
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{Euclidean, Manifold, Sphere};
    use crate::trust::dogleg_direction;
    use ndarray::{array, Array2, ArrayView1};

    #[test]
    fn tcg_takes_the_newton_point_inside_a_large_radius() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let (p, drop) = tcg(&g, 10.0, 8, |v| h.dot(v));
        assert!((p[0] + 1.0).abs() < 1e-10);
        assert!(p[1].abs() < 1e-10);
        assert!(drop > 0.0);
    }

    #[test]
    fn tcg_stays_on_the_trust_sphere() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let (p, _) = tcg(&g, 0.1, 8, |v| h.dot(v));
        let n = nrm2(p.view());
        assert!((n - 0.1).abs() < 1e-12);
        assert!(p[0] < 0.0);
    }

    #[test]
    fn tcg_negative_curvature_walks_to_the_boundary() {
        let h = Array2::<f64>::from_shape_vec((2, 2), vec![-2.0, 0.0, 0.0, 1.0]).unwrap();
        let g = array![0.1, 0.0];
        let (p, _) = tcg(&g, 0.5, 8, |v| h.dot(v));
        assert!((nrm2(p.view()) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn rhess_fd_matches_a_constant_euclidean_hessian() {
        let obj = crate::Oracle::unbounded(2, |x: ArrayView1<f64>| {
            (5.0 * x[0] * x[0] + 0.5 * x[1] * x[1], array![10.0 * x[0], x[1]])
        });
        let x = array![0.3, -0.4];
        let rgrad = array![3.0, -0.4];
        let eta = array![0.2, 0.1];
        let hv = rhess_fd(&obj, &Euclidean, &x, &rgrad, &eta, 1e-6);
        assert!((hv[0] - 2.0).abs() < 1e-6);
        assert!((hv[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn rhess_fd_on_the_sphere_is_tangent() {
        let obj = crate::Oracle::unbounded(3, |x: ArrayView1<f64>| {
            (
                0.5 * (x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2]),
                array![x[0], 2.0 * x[1], 3.0 * x[2]],
            )
        });
        let n = (3.0_f64).sqrt();
        let x = array![1.0 / n, 1.0 / n, 1.0 / n];
        let egrad = array![x[0], 2.0 * x[1], 3.0 * x[2]];
        let rgrad = Sphere.egrad2rgrad(&x, &egrad);
        let raw = array![0.2, -0.1, 0.05];
        let eta = Sphere.project(&x, &raw);
        let hv = rhess_fd(&obj, &Sphere, &x, &rgrad, &eta, 1e-6);
        assert!(dot(x.view(), hv.view()).abs() < 1e-8, "Hess eta left the tangent");
    }

    #[test]
    fn tcg_and_dogleg_agree_on_a_tight_euclidean_model() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let (p_tcg, _) = tcg(&g, 0.1, 8, |v| h.dot(v));
        let p_dog = dogleg_direction(&h, &g, 0.1);
        assert!((nrm2(p_tcg.view()) - nrm2(p_dog.view())).abs() < 1e-12);
        assert!((p_tcg[0] - p_dog[0]).abs() < 1e-12);
    }
}
