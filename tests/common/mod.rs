//! Morse and LJ oracles, the `.con` reader and run helpers shared by the
//! dogleg and Newton-CG campaigns.

#![allow(dead_code)]

use std::path::PathBuf;

use ndarray::{Array1, ArrayView1};
use rgmin::Control;

pub const MORSE_D0: f64 = 0.7102;
pub const MORSE_ALPHA: f64 = 1.6047;
pub const MORSE_R0: f64 = 2.897;
pub const MORSE_CUTOFF: f64 = 7.0;

pub const LJ_EPSILON: f64 = 1.0;
pub const LJ_SIGMA: f64 = 1.0;
pub const LJ_CUTOFF: f64 = 40.0;

pub fn min_image(d: f64, length: f64) -> f64 {
    d - length * (d / length).round()
}

pub fn morse_pairs(pos: ArrayView1<f64>, box_l: [f64; 3]) -> Vec<(usize, usize, [f64; 3], f64)> {
    let n = pos.len() / 3;
    let cut2 = MORSE_CUTOFF * MORSE_CUTOFF;
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                let d = min_image(pos[3 * i + a] - pos[3 * j + a], box_l[a]);
                diff[a] = d;
                r2 += d * d;
            }
            if r2 > 0.0 && r2 < cut2 {
                pairs.push((i, j, diff, r2));
            }
        }
    }
    pairs
}

pub fn morse_value_grad(pos: ArrayView1<f64>, box_l: [f64; 3]) -> (f64, Array1<f64>) {
    let pairs = morse_pairs(pos, box_l);
    let rc = MORSE_CUTOFF;
    let xc = (-MORSE_ALPHA * (rc - MORSE_R0)).exp();
    let shift = MORSE_D0 * (xc * xc - 2.0 * xc);
    let mut energy = 0.0;
    let mut grad = Array1::<f64>::zeros(pos.len());
    for (i, j, diff, r2) in pairs {
        let r = r2.sqrt();
        let x = (-MORSE_ALPHA * (r - MORSE_R0)).exp();
        energy += MORSE_D0 * (x * x - 2.0 * x) - shift;
        let slope = (2.0 * MORSE_D0 * MORSE_ALPHA * x * (1.0 - x)) / r;
        for a in 0..3 {
            let piece = slope * diff[a];
            grad[3 * i + a] += piece;
            grad[3 * j + a] -= piece;
        }
    }
    (energy, grad)
}

pub fn lj_value_grad(pos: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = pos.len() / 3;
    let cut2 = LJ_CUTOFF * LJ_CUTOFF;
    let mut energy = 0.0;
    let mut grad = Array1::<f64>::zeros(pos.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let mut diff = [0.0; 3];
            let mut r2 = 0.0;
            for a in 0..3 {
                let d = pos[3 * i + a] - pos[3 * j + a];
                diff[a] = d;
                r2 += d * d;
            }
            if r2 <= 0.0 || r2 >= cut2 {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let sr2 = (LJ_SIGMA * LJ_SIGMA) * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            energy += 4.0 * LJ_EPSILON * (sr12 - sr6);
            let slope = 24.0 * LJ_EPSILON * (sr6 - 2.0 * sr12) * inv_r2;
            for a in 0..3 {
                let piece = slope * diff[a];
                grad[3 * i + a] += piece;
                grad[3 * j + a] -= piece;
            }
        }
    }
    (energy, grad)
}

pub fn read_con(text: &str) -> (Array1<f64>, [f64; 3]) {
    let lines: Vec<&str> = text.lines().collect();
    let mut box_l = None;
    for i in 0..lines.len().saturating_sub(1) {
        let parts: Vec<f64> = lines[i]
            .split_whitespace()
            .filter_map(|p| p.parse().ok())
            .collect();
        if parts.len() != 3 {
            continue;
        }
        let angles: Vec<f64> = lines[i + 1]
            .split_whitespace()
            .filter_map(|p| p.parse().ok())
            .collect();
        if angles.len() == 3 && angles.iter().all(|a| (a - 90.0).abs() < 1e-4) {
            box_l = Some([parts[0], parts[1], parts[2]]);
            break;
        }
    }
    let start = lines
        .iter()
        .position(|line| line.starts_with("Coordinates of Component"))
        .expect("con coordinates");
    let mut coords = Vec::new();
    for line in &lines[start + 1..] {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            break;
        }
        let (Ok(x), Ok(y), Ok(z)) = (
            parts[0].parse::<f64>(),
            parts[1].parse::<f64>(),
            parts[2].parse::<f64>(),
        ) else {
            break;
        };
        coords.extend([x, y, z]);
    }
    (Array1::from(coords), box_l.expect("orthorhombic box"))
}

pub fn load_fixture(name: &str) -> (Array1<f64>, [f64; 3]) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"));
    read_con(&text)
}

pub fn max_atom_step(before: &Array1<f64>, after: &Array1<f64>) -> f64 {
    let n = before.len();
    let mut m = 0.0_f64;
    let mut i = 0;
    while i + 3 <= n {
        let dx = after[i] - before[i];
        let dy = after[i + 1] - before[i + 1];
        let dz = after[i + 2] - before[i + 2];
        m = m.max((dx * dx + dy * dy + dz * dz).sqrt());
        i += 3;
    }
    m
}

pub fn euclid(v: &Array1<f64>) -> f64 {
    v.iter().map(|t| t * t).sum::<f64>().sqrt()
}

pub fn control(gtol: f64, maxiter: usize) -> Control {
    Control::default()
        .with_maxiter(maxiter)
        .with_gtol(gtol)
        .with_istep(0.1)
        .with_maxmove(None)
}

pub fn worker_count(n: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    cores.min(n.max(1))
}
