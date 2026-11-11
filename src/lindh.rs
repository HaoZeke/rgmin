//! Sparse Lindh model Hessian, used as `H₀⁻¹ = P⁻¹`.
//!
//! Stretch and bend constants are Lindh, Bernhardsson, Karlström, and
//! Malmqvist, Chem. Phys. Lett. 241, 423 (1995),
//! <https://doi.org/10.1016/0009-2614(95)00646-L>, as tabulated by
//! Bakken and Helgaker, J. Chem. Phys. 117, 9160 (2002),
//! <https://doi.org/10.1063/1.1515483>. The Cartesian block of a
//! coordinate `ξ` is `k (∂ξ/∂x)(∂ξ/∂x)ᵀ`, which is the Hessian of a
//! quadratic whose equilibrium is the current geometry. Torsions are
//! left out: `kτ = 0.005` is two orders below the stretch.
//!
//! `ρᵢⱼ = exp(αᵢⱼ (r_ref² − r²))` with `r` in bohr. Rows are the
//! periodic-table periods in that table; an atomic number past argon
//! uses the third row. A positive `c_stab I` makes `P` definite.
//! `μ` scales the whole matrix into the objective's units.
//!
//! The factor is a banded Cholesky after a reverse Cuthill–McKee
//! ordering of the atoms. A bandwidth whose factor would exceed
//! about `8×10⁸` flops returns [`None`] and the caller keeps the
//! scalar initial Hessian.

use std::collections::BTreeMap;

/// Å to bohr. `ρ` is defined on the bohr length in the Lindh table.
const BOHR_PER_ANGSTROM: f64 = 1.889_726_124_626;

const ALPHA: [[f64; 3]; 3] = [
    [1.0, 0.3949, 0.3949],
    [0.3949, 0.28, 0.28],
    [0.3949, 0.28, 0.28],
];

const RREF: [[f64; 3]; 3] = [[1.35, 2.10, 2.53], [2.10, 2.87, 3.40], [2.53, 3.40, 3.40]];

const K_STRETCH: f64 = 0.45;
const K_BEND: f64 = 0.15;

/// Drop stretches softer than this. The bend list uses a higher cut.
const STRETCH_RHO: f64 = 1.0e-4;
const BEND_RHO: f64 = 0.05;

/// Atomic numbers and the ridge used to build [`factor`].
#[derive(Clone, Debug)]
pub struct LindhSpec {
    /// Length `N`. Empty means every atom is period 1 (H, He).
    pub z: Vec<u8>,
    /// Diagonal ridge in the unscaled matrix. Must be positive.
    pub c_stab: f64,
    /// Periodic lengths. A non-positive entry is open.
    pub cell: [f64; 3],
}

/// Banded Cholesky of `μ P`.
#[derive(Clone, Debug)]
pub(crate) struct LindhFactor {
    n: usize,
    band: usize,
    /// Row-major lower band, including the diagonal.
    l: Vec<f64>,
    /// `perm[new] = old` coordinate index.
    perm: Vec<usize>,
    /// `iperm[old] = new`.
    iperm: Vec<usize>,
}

/// `vᵀ P v` at `μ = 1`, including `c_stab`.
#[must_use]
pub(crate) fn shape_quad(x: &[f64], z: &[u8], c_stab: f64, cell: [f64; 3], v: &[f64]) -> f64 {
    if x.len() != v.len() || c_stab <= 0.0 {
        return 0.0;
    }
    let acc = assemble(x, z, c_stab, cell);
    let mut q = 0.0;
    for ((i, j), a) in acc {
        if i == j {
            q += a * v[i] * v[i];
        } else {
            q += 2.0 * a * v[i] * v[j];
        }
    }
    q
}

/// Factor `μ P`. `None` when the matrix is indefinite, the dimension
/// is not a multiple of 3, or the band is too wide to factor here.
#[must_use]
pub(crate) fn factor(
    x: &[f64],
    z: &[u8],
    c_stab: f64,
    cell: [f64; 3],
    mu: f64,
) -> Option<LindhFactor> {
    if !(mu.is_finite() && mu > 0.0 && c_stab.is_finite() && c_stab > 0.0) {
        return None;
    }
    let n = x.len();
    if n == 0 || n % 3 != 0 {
        return None;
    }
    let acc = assemble(x, z, c_stab, cell);
    let natoms = n / 3;
    let mut adj = vec![Vec::new(); natoms];
    for &(i, j) in acc.keys() {
        if i != j {
            let ai = i / 3;
            let aj = j / 3;
            if ai != aj {
                adj[ai].push(aj);
                adj[aj].push(ai);
            }
        }
    }
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    let order = rcm(natoms, &adj);
    let mut iperm_atom = vec![0; natoms];
    for (new, &old) in order.iter().enumerate() {
        iperm_atom[old] = new;
    }
    let mut perm = vec![0; n];
    let mut iperm = vec![0; n];
    for old in 0..n {
        let new = 3 * iperm_atom[old / 3] + (old % 3);
        perm[new] = old;
        iperm[old] = new;
    }
    let mut band = 0usize;
    for &(i, j) in acc.keys() {
        let d = iperm[i].abs_diff(iperm[j]);
        band = band.max(d);
    }
    if n as u128 * (band as u128 + 1) * (band as u128 + 1) > 800_000_000 {
        return None;
    }
    let width = band + 1;
    let mut a = vec![0.0; n * width];
    for ((i, j), val) in acc {
        let pi = iperm[i];
        let pj = iperm[j];
        let (row, col) = if pi >= pj { (pi, pj) } else { (pj, pi) };
        let off = col + band - row;
        a[row * width + off] += mu * val;
    }
    for i in 0..n {
        for j in i.saturating_sub(band)..i {
            let mut sum = a[i * width + (j + band - i)];
            let k0 = i.saturating_sub(band).max(j.saturating_sub(band));
            for k in k0..j {
                sum -= a[i * width + (k + band - i)] * a[j * width + (k + band - j)];
            }
            a[i * width + (j + band - i)] = sum / a[j * width + band];
        }
        let mut sum = a[i * width + band];
        let k0 = i.saturating_sub(band);
        for k in k0..i {
            let lik = a[i * width + (k + band - i)];
            sum -= lik * lik;
        }
        if !(sum.is_finite() && sum > 1e-14) {
            return None;
        }
        a[i * width + band] = sum.sqrt();
    }
    Some(LindhFactor {
        n,
        band,
        l: a,
        perm,
        iperm,
    })
}

impl LindhFactor {
    /// `P⁻¹ rhs`. False when the lengths differ.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn solve_into(&self, q: &mut ndarray::Array1<f64>) -> bool {
        let n = self.n;
        if q.len() != n {
            return false;
        }
        let src: Vec<f64> = q.iter().copied().collect();
        let mut rhs = vec![0.0; n];
        for old in 0..n {
            rhs[self.iperm[old]] = src[old];
        }
        let width = self.band + 1;
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = rhs[i];
            let j0 = i.saturating_sub(self.band);
            for j in j0..i {
                sum -= self.l[i * width + (j + self.band - i)] * y[j];
            }
            y[i] = sum / self.l[i * width + self.band];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            let j1 = (i + self.band).min(n - 1);
            for j in (i + 1)..=j1 {
                sum -= self.l[j * width + (i + self.band - j)] * x[j];
            }
            x[i] = sum / self.l[i * width + self.band];
        }
        if x.iter().any(|v| !v.is_finite()) {
            return false;
        }
        for new in 0..n {
            q[self.perm[new]] = x[new];
        }
        true
    }
}

fn assemble(x: &[f64], z: &[u8], c_stab: f64, cell: [f64; 3]) -> BTreeMap<(usize, usize), f64> {
    let mut acc = BTreeMap::new();
    let n = x.len();
    let natoms = n / 3;
    if natoms == 0 || n != 3 * natoms {
        return acc;
    }
    for i in 0..n {
        acc.insert((i, i), c_stab);
    }
    let mut bonds: Vec<Vec<(usize, f64)>> = vec![Vec::new(); natoms];
    for i in 0..natoms {
        for j in (i + 1)..natoms {
            let d = delta(x, i, j, cell);
            let r_ang = norm(&d);
            if r_ang <= 1e-8 {
                continue;
            }
            let ri = row_of(z, i);
            let rj = row_of(z, j);
            let r_bohr = r_ang * BOHR_PER_ANGSTROM;
            let alpha = ALPHA[ri][rj];
            let rref = RREF[ri][rj];
            let rho = (alpha * (rref * rref - r_bohr * r_bohr)).exp();
            if !(rho.is_finite() && rho > STRETCH_RHO) {
                continue;
            }
            let u = [d[0] / r_ang, d[1] / r_ang, d[2] / r_ang];
            let k = K_STRETCH * rho * BOHR_PER_ANGSTROM * BOHR_PER_ANGSTROM;
            let idx = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2];
            let g = [-u[0], -u[1], -u[2], u[0], u[1], u[2]];
            add_rank1(&mut acc, &idx, &g, k);
            if rho > BEND_RHO {
                bonds[i].push((j, rho));
                bonds[j].push((i, rho));
            }
        }
    }
    for (j, nbrs) in bonds.iter().enumerate() {
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                let (i, rho_ij) = nbrs[a];
                let (k, rho_jk) = nbrs[b];
                add_angle(&mut acc, x, cell, i, j, k, K_BEND * rho_ij * rho_jk);
            }
        }
    }
    acc
}

fn add_angle(
    acc: &mut BTreeMap<(usize, usize), f64>,
    x: &[f64],
    cell: [f64; 3],
    i: usize,
    j: usize,
    k: usize,
    kconst: f64,
) {
    if !(kconst.is_finite() && kconst > 0.0) {
        return;
    }
    let a = delta(x, j, i, cell);
    let b = delta(x, j, k, cell);
    let la = norm(&a);
    let lb = norm(&b);
    if la < 1e-8 || lb < 1e-8 {
        return;
    }
    let c = (dot3(&a, &b) / (la * lb)).clamp(-1.0 + 1e-12, 1.0 - 1e-12);
    let sin = (1.0 - c * c).sqrt();
    if sin < 1e-6 {
        return;
    }
    let inv = 1.0 / (la * lb);
    let mut gi = [0.0; 3];
    let mut gk = [0.0; 3];
    for t in 0..3 {
        let dc_i = b[t] * inv - c * a[t] / (la * la);
        let dc_k = a[t] * inv - c * b[t] / (lb * lb);
        gi[t] = -dc_i / sin;
        gk[t] = -dc_k / sin;
    }
    let gj = [-gi[0] - gk[0], -gi[1] - gk[1], -gi[2] - gk[2]];
    let idx = [
        3 * i,
        3 * i + 1,
        3 * i + 2,
        3 * j,
        3 * j + 1,
        3 * j + 2,
        3 * k,
        3 * k + 1,
        3 * k + 2,
    ];
    let g = [
        gi[0], gi[1], gi[2], gj[0], gj[1], gj[2], gk[0], gk[1], gk[2],
    ];
    add_rank1(acc, &idx, &g, kconst);
}

fn add_rank1(acc: &mut BTreeMap<(usize, usize), f64>, idx: &[usize], g: &[f64], k: f64) {
    for a in 0..idx.len() {
        for b in 0..=a {
            let v = k * g[a] * g[b];
            if v == 0.0 {
                continue;
            }
            let (r, c) = if idx[a] >= idx[b] {
                (idx[a], idx[b])
            } else {
                (idx[b], idx[a])
            };
            *acc.entry((r, c)).or_insert(0.0) += v;
        }
    }
}

fn row_of(z: &[u8], atom: usize) -> usize {
    if atom < z.len() {
        match z[atom] {
            0..=2 => 0,
            3..=10 => 1,
            _ => 2,
        }
    } else {
        0
    }
}

fn delta(x: &[f64], i: usize, j: usize, cell: [f64; 3]) -> [f64; 3] {
    let mut d = [0.0; 3];
    for a in 0..3 {
        let mut v = x[3 * j + a] - x[3 * i + a];
        if cell[a] > 0.0 {
            v -= cell[a] * (v / cell[a]).round();
        }
        d[a] = v;
    }
    d
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn rcm(n: usize, adj: &[Vec<usize>]) -> Vec<usize> {
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut left = n;
    while left > 0 {
        let mut start = 0;
        let mut best = usize::MAX;
        for i in 0..n {
            if !seen[i] && adj[i].len() < best {
                best = adj[i].len();
                start = i;
            }
        }
        let end = bfs_far(start, adj, &seen);
        let mut q = std::collections::VecDeque::new();
        q.push_back(end);
        seen[end] = true;
        left -= 1;
        while let Some(u) = q.pop_front() {
            order.push(u);
            let mut nbrs = Vec::new();
            for &v in &adj[u] {
                if !seen[v] {
                    nbrs.push(v);
                }
            }
            nbrs.sort_by_key(|v| adj[*v].len());
            for v in nbrs {
                if !seen[v] {
                    seen[v] = true;
                    left -= 1;
                    q.push_back(v);
                }
            }
        }
    }
    order.reverse();
    order
}

fn bfs_far(start: usize, adj: &[Vec<usize>], blocked: &[bool]) -> usize {
    let mut seen = blocked.to_vec();
    let mut q = std::collections::VecDeque::new();
    q.push_back(start);
    seen[start] = true;
    let mut last = start;
    while let Some(u) = q.pop_front() {
        last = u;
        for &v in &adj[u] {
            if !seen[v] {
                seen[v] = true;
                q.push_back(v);
            }
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(acc: &BTreeMap<(usize, usize), f64>, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; v.len()];
        for ((i, j), a) in acc {
            out[*i] += a * v[*j];
            if i != j {
                out[*j] += a * v[*i];
            }
        }
        out
    }

    #[test]
    fn the_factor_reproduces_a_dense_product() {
        let x = [0.0, 0.0, 0.0, 0.74, 0.0, 0.0, 0.2, 0.9, 0.05];
        let z = [1u8, 1, 8];
        let c = 0.1;
        let cell = [0.0; 3];
        let acc = assemble(&x, &z, c, cell);
        let factor = factor(&x, &z, c, cell, 1.7).unwrap();
        let rhs = [0.2, -0.1, 0.05, 0.3, 0.0, -0.2, 0.1, 0.4, -0.3];
        let mut q = ndarray::Array1::from(rhs.to_vec());
        assert!(factor.solve_into(&mut q));
        let pv = apply(&acc, q.as_slice().unwrap());
        for i in 0..9 {
            let got = 1.7 * pv[i];
            assert!((got - rhs[i]).abs() < 1e-7, "{i}: {got} vs {}", rhs[i]);
        }
    }

    #[test]
    fn a_bond_is_stiffer_than_a_perpendicular_shift() {
        let x = [0.0, 0.0, 0.0, 0.74, 0.0, 0.0];
        let along = [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let perp = [0.0, -1.0, 0.0, 0.0, 1.0, 0.0];
        let qa = shape_quad(&x, &[1, 1], 0.05, [0.0; 3], &along);
        let qp = shape_quad(&x, &[1, 1], 0.05, [0.0; 3], &perp);
        assert!(qa > qp * 2.0, "bond {qa} perp {qp}");
    }

    #[test]
    fn a_lone_atom_solves_as_the_ridge() {
        let x = [0.0, 0.0, 0.0];
        let f = factor(&x, &[1], 2.0, [0.0; 3], 1.0).unwrap();
        let mut q = ndarray::Array1::from(vec![2.0, 0.0, -2.0]);
        assert!(f.solve_into(&mut q));
        assert!((q[0] - 1.0).abs() < 1e-10, "{}", q[0]);
        assert!(q[1].abs() < 1e-10);
        assert!((q[2] + 1.0).abs() < 1e-10, "{}", q[2]);
    }
}
