//! SO(3) as a 9-vector, row-major. manopt_cpp `Rotation<3,1>`.
//!
//! Dimension is exactly 9. A 3N cluster is
//! [`super::RigidQuotient`], not this embedding.

use ndarray::Array1;

use super::Manifold;

/// Rotation matrices packed row-major length 9.
#[derive(Clone, Copy, Debug, Default)]
pub struct So3;

fn unpack(x: &Array1<f64>) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    if x.len() >= 9 {
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = x[3 * i + j];
            }
        }
    }
    r
}

fn pack(r: [[f64; 3]; 3]) -> Array1<f64> {
    Array1::from_shape_vec(9, {
        let mut v = Vec::with_capacity(9);
        for row in &r {
            v.extend_from_slice(row);
        }
        v
    })
    .unwrap()
}

fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn skew_of(s: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut k = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            k[i][j] = 0.5 * (s[i][j] - s[j][i]);
        }
    }
    k
}

/// Polar-ish retraction: SVD-free QR with positive diagonal.
fn qr_pos(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Gram-Schmidt columns.
    let mut q = [[0.0; 3]; 3];
    for j in 0..3 {
        let mut v = [a[0][j], a[1][j], a[2][j]];
        #[allow(
            clippy::needless_range_loop,
            reason = "k picks a column of q, and the suggested q.iter() walks rows"
        )]
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..3 {
                dot += q[i][k] * a[i][j];
            }
            for i in 0..3 {
                v[i] -= dot * q[i][k];
            }
        }
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 1e-16 {
            for i in 0..3 {
                q[i][j] = v[i] / n;
            }
        }
    }
    let det = q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1])
        - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0])
        + q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0]);
    if det < 0.0 {
        for row in &mut q {
            row[2] = -row[2];
        }
    }
    q
}

impl Manifold for So3 {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == 9 { Ok(()) } else { Err(9) }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 9 || v.len() != 9 {
            return v.clone();
        }
        let r = unpack(x);
        let h = unpack(v);
        let rt_h = mul(transpose(r), h);
        let omega = skew_of(rt_h);
        pack(mul(r, omega))
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() != 9 || v.len() != 9 {
            return x + v;
        }
        let r = unpack(x);
        let u = unpack(v);
        let mut y = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                y[i][j] = r[i][j] + u[i][j];
            }
        }
        pack(qr_pos(y))
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
        if [x, egrad, ehess, v].iter().any(|a| a.len() != 9) {
            return None;
        }
        let r = unpack(x);
        let rt = transpose(r);
        let omega = skew_of(mul(rt, unpack(v)));
        let sym_g = sym_of(mul(rt, unpack(egrad)));
        let rt_h = mul(rt, unpack(ehess));
        let bent = mul(omega, sym_g);
        let mut inner = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                inner[i][j] = rt_h[i][j] - bent[i][j];
            }
        }
        Some(pack(mul(r, skew_of(inner))))
    }

    fn exp(&self, x: &Array1<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
        if x.len() != 9 || v.len() != 9 {
            return None;
        }
        let r = unpack(x);
        let omega = skew_of(mul(transpose(r), unpack(v)));
        Some(pack(mul(r, expm_skew(omega))))
    }

    fn log(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<Array1<f64>> {
        if x.len() != 9 || y.len() != 9 {
            return None;
        }
        let r = unpack(x);
        Some(pack(mul(r, logm_rotation(mul(transpose(r), unpack(y))))))
    }

    fn dist(&self, x: &Array1<f64>, y: &Array1<f64>) -> Option<f64> {
        if x.len() != 9 || y.len() != 9 {
            return None;
        }
        let omega = logm_rotation(mul(transpose(unpack(x)), unpack(y)));
        Some(frobenius(omega))
    }

    fn parallel_transport_along(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Option<Array1<f64>> {
        if x.len() != 9 || d.len() != 9 || v.len() != 9 {
            return None;
        }
        let r = unpack(x);
        let rt = transpose(r);
        let mut half = skew_of(mul(rt, unpack(d)));
        for row in &mut half {
            for e in row.iter_mut() {
                *e *= 0.5;
            }
        }
        let e = expm_skew(half);
        let moved = mul(e, mul(skew_of(mul(rt, unpack(v))), e));
        Some(pack(mul(r, moved)))
    }

    fn typical_dist(&self, _n: usize) -> f64 {
        std::f64::consts::PI * 3.0_f64.sqrt()
    }
}

fn sym_of(s: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut k = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            k[i][j] = 0.5 * (s[i][j] + s[j][i]);
        }
    }
    k
}

fn frobenius(a: [[f64; 3]; 3]) -> f64 {
    a.iter().flatten().map(|e| e * e).sum::<f64>().sqrt()
}

/// Axis-angle vector \(\omega\) of a skew matrix, \(\Omega u = \omega \times u\).
fn vee(omega: [[f64; 3]; 3]) -> [f64; 3] {
    [omega[2][1], omega[0][2], omega[1][0]]
}

fn hat(w: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

/// Rodrigues: \(e^\Omega = I + \frac{\sin\theta}{\theta}\Omega
/// + \frac{1-\cos\theta}{\theta^2}\Omega^2\), \(\theta = \|\omega\|\).
fn expm_skew(omega: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let w = vee(omega);
    let t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let t = t2.sqrt();
    let (a, b) = if t < 1e-4 {
        (1.0 - t2 / 6.0, 0.5 - t2 / 24.0)
    } else {
        (t.sin() / t, (1.0 - t.cos()) / t2)
    };
    let sq = mul(omega, omega);
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            out[i][j] = id + a * omega[i][j] + b * sq[i][j];
        }
    }
    out
}

/// Principal logarithm of a rotation, the skew \(\Omega\) with
/// \(\|\omega\| = \theta \in [0, \pi]\). Near \(\theta = \pi\) the axis
/// comes from the symmetric part \((1-\cos\theta)\,n n^\top\), where the
/// skew part \(\sin\theta\,\hat n\) has lost its digits.
fn logm_rotation(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let c = (0.5 * (trace - 1.0)).clamp(-1.0, 1.0);
    let w = vee(skew_of(r));
    let s = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    let theta = s.atan2(c);
    if c > -0.9 {
        let k = if s < 1e-12 { 1.0 } else { theta / s };
        return hat([k * w[0], k * w[1], k * w[2]]);
    }
    let sym = sym_of(r);
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = sym[i][j] - if i == j { c } else { 0.0 };
        }
    }
    let k = (0..3)
        .max_by(|&p, &q| b[p][p].total_cmp(&b[q][q]))
        .unwrap_or(0);
    let scale = (b[k][k] * (1.0 - c)).max(f64::MIN_POSITIVE).sqrt();
    let mut n = [b[0][k] / scale, b[1][k] / scale, b[2][k] / scale];
    if n[0] * w[0] + n[1] * w[1] + n[2] * w[2] < 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }
    hat([theta * n[0], theta * n[1], theta * n[2]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_plus_skew_stays_orthogonal() {
        let x = array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = array![0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y = So3.retract(&x, &v);
        let r = unpack(&y);
        let rtr = mul(transpose(r), r);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - want).abs() < 1e-12, "{rtr:?}");
            }
        }
    }

    fn rotation(w: [f64; 3]) -> Array1<f64> {
        pack(expm_skew(hat(w)))
    }

    fn eye() -> Array1<f64> {
        array![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    #[test]
    fn exp_log_round_trip_up_to_pi() {
        let x = rotation([0.3, -0.2, 0.5]);
        for (axis, theta) in [
            ([0.0, 0.0, 1.0], 1e-7),
            ([0.6, 0.0, 0.8], 0.9),
            ([0.0, 0.6, 0.8], 2.95),
            ([0.48, 0.6, 0.64], std::f64::consts::PI - 1e-6),
        ] {
            let v = pack(mul(unpack(&x), hat(axis.map(|a| a * theta))));
            let y = So3.exp(&x, &v).unwrap();
            let back = So3.log(&x, &y).unwrap();
            let err = (&back - &v).mapv(f64::abs).fold(0.0_f64, |m, &e| m.max(e));
            assert!(err < 1e-8, "theta {theta}: {err}");
            let d = So3.dist(&x, &y).unwrap();
            assert!((d - std::f64::consts::SQRT_2 * theta).abs() < 1e-9);
        }
    }

    #[test]
    fn log_handles_a_half_turn() {
        let y = rotation([0.0, std::f64::consts::PI, 0.0]);
        let omega = So3.log(&eye(), &y).unwrap();
        let back = So3.exp(&eye(), &omega).unwrap();
        let err = (&back - &y).mapv(f64::abs).fold(0.0_f64, |m, &e| m.max(e));
        assert!(err < 1e-12, "{err}");
    }

    #[test]
    fn parallel_transport_is_isometric_and_tangent() {
        let x = rotation([0.1, 0.4, -0.3]);
        let d = So3.project(&x, &array![0.3, -0.1, 0.2, 0.5, 0.1, -0.4, 0.2, 0.3, 0.1]);
        let v = So3.project(&x, &array![-0.2, 0.4, 0.1, 0.0, 0.3, 0.2, -0.1, 0.5, 0.2]);
        let y = So3.exp(&x, &d).unwrap();
        let moved = So3.parallel_transport_along(&x, &d, &v).unwrap();
        assert!((moved.dot(&moved) - v.dot(&v)).abs() < 1e-12);
        let off = &moved - &So3.project(&y, &moved);
        assert!(off.dot(&off).sqrt() < 1e-12);
        let velocity = So3.parallel_transport_along(&x, &d, &d).unwrap();
        let log_back = So3.log(&y, &x).unwrap();
        let sum = &velocity + &log_back;
        assert!(
            sum.dot(&sum).sqrt() < 1e-10,
            "geodesic velocity is not parallel"
        );
    }

    #[test]
    fn wrong_dim_does_not_shrink() {
        let x = Array1::from_elem(114, 0.1);
        let v = Array1::from_elem(114, 0.01);
        let y = So3.retract(&x, &v);
        assert_eq!(y.len(), 114);
        assert_eq!(So3.project(&x, &v).len(), 114);
        assert!(So3.required_dim(114).is_err());
        assert!(So3.required_dim(9).is_ok());
    }
}
