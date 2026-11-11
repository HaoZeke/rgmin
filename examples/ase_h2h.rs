// Copyright (c) rgmin developers

//! rgmin side of the ASE head-to-head.
//!
//! Four arms, one force tolerance and one step cap, on every system:
//! capped L-BFGS, Wolfe L-BFGS, the geometry step, and FIRE 2.0.
//! The cap is 0.2. Capped L-BFGS and the geometry step apply it per
//! atom, which is ASE LBFGS `maxstep`. FIRE 2.0 applies it to the
//! Euclidean length of the displacement, which is ASE FIRE `maxstep`.
//! The Wolfe search has no displacement cap: a per-atom cap on that
//! session selects the geometry step instead.
//! Memory is 8, the crate default, on every L-BFGS arm. FIRE 2.0
//! starts at timestep 0.1, ASE FIRE's default, with the crate's own
//! mixing parameters. The stop is the largest atomic force, 0.01.
//! Seven serial repeats. The printed wall time is the median.
//!
//! `benches/ase_h2h.py` runs this binary and the ASE arms.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgmin::{Accept, Control, FireKind, LbfgsStep, Method, Oracle, Solver, StopNorm};

const REPEATS: usize = 7;
const BUDGET: usize = 100_000;
const FMAX: f64 = 0.01;
const MAXSTEP: f64 = 0.2;
const FIRE_DT: f64 = 0.1;
const SPACING: f64 = 1.1;
const JITTER: f64 = 0.03;

#[derive(Clone, Copy)]
enum Arm {
    Wolfe,
    Capped,
    Geometry,
    Fire2,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Wolfe => "rgmin-wolfe",
            Self::Capped => "rgmin-capped",
            Self::Geometry => "rgmin-geometry",
            Self::Fire2 => "rgmin-fire2",
        }
    }
}

const ARMS: [Arm; 4] = [Arm::Wolfe, Arm::Capped, Arm::Geometry, Arm::Fire2];

fn lennard_jones(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let s6 = r2.powi(-3);
            e += 4.0 * (s6 * s6 - s6);
            let du_over_r = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for a in 0..3 {
                g[3 * i + a] += du_over_r * d[a];
                g[3 * j + a] -= du_over_r * d[a];
            }
        }
    }
    (e, g)
}

fn morse_pair(r: f64) -> (f64, f64) {
    let (d0, a, r0): (f64, f64, f64) = (0.7102, 1.6047, 2.897);
    let xc = (-a * (7.0 - r0)).exp();
    let x = (-a * (r - r0)).exp();
    (
        d0 * (x * x - 2.0 * x) - d0 * (xc * xc - 2.0 * xc),
        2.0 * d0 * a * x * (1.0 - x),
    )
}

fn morse(cell: [f64; 3], x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [0.0; 3];
            for a in 0..3 {
                d[a] = x[3 * i + a] - x[3 * j + a];
                d[a] -= cell[a] * (d[a] / cell[a]).round();
            }
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r >= 7.0 {
                continue;
            }
            let (u, du) = morse_pair(r);
            e += u;
            for a in 0..3 {
                g[3 * i + a] += du * d[a] / r;
                g[3 * j + a] -= du * d[a] / r;
            }
        }
    }
    (e, g)
}

fn grid_start(atoms: usize) -> Array1<f64> {
    let side = (atoms as f64).cbrt().ceil() as usize;
    let mut x = Vec::with_capacity(3 * atoms);
    for k in 0..atoms {
        let cell = [k % side, (k / side) % side, k / (side * side)];
        for (a, c) in cell.iter().enumerate() {
            x.push(SPACING * *c as f64 + JITTER * ((3 * k + a) as f64).sin());
        }
    }
    Array1::from(x)
}

fn read_con(name: &str) -> (Array1<f64>, [f64; 3]) {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture");
    let lines: Vec<&str> = text.lines().collect();
    let num = |l: &str| -> Vec<f64> {
        l.split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect()
    };
    let cell = num(lines[2]);
    let start = lines
        .iter()
        .position(|l| l.starts_with("Coordinates"))
        .unwrap();
    let mut x = Vec::new();
    for l in &lines[start + 1..] {
        let v = num(l);
        if v.len() < 3 {
            break;
        }
        x.extend(&v[..3]);
    }
    (Array1::from(x), [cell[0], cell[1], cell[2]])
}

fn fmax_of(g: &Array1<f64>) -> f64 {
    StopNorm::MaxAtom.of(g.view())
}

struct Outcome {
    evals: usize,
    energy: f64,
    fmax: f64,
    ms: f64,
    converged: bool,
}

fn once<F>(arm: Arm, x0: &Array1<f64>, fg: F) -> (usize, f64, f64, bool)
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync + Copy,
{
    let calls = AtomicUsize::new(0);
    let obj = Oracle::unbounded(x0.len(), |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        fg(x)
    });
    let control = Control {
        maxiter: BUDGET,
        gtol: FMAX,
        istep: match arm {
            Arm::Fire2 => FIRE_DT,
            _ => 1.0,
        },
        stop_norm: StopNorm::MaxAtom,
        ..Control::default()
    };
    let method = match arm {
        Arm::Fire2 => Method::Fire { kind: FireKind::V2 },
        _ => Method::Lbfgs { memory: 8 },
    };
    let mut solver = Solver::new(method, control, x0.len());
    match arm {
        Arm::Wolfe => {
            solver.set_lbfgs_step(LbfgsStep::Wolfe).expect("wolfe");
            solver.set_accept(Accept::Energy);
        }
        Arm::Capped => solver
            .set_lbfgs_step(LbfgsStep::Capped { max_atom: MAXSTEP })
            .expect("capped step"),
        Arm::Geometry => {
            solver.set_lbfgs_step(LbfgsStep::Wolfe).expect("wolfe");
            solver.set_atom_maxmove(MAXSTEP);
        }
        Arm::Fire2 => solver.set_maxmove(MAXSTEP),
    }
    let mut x = x0.clone();
    let mut converged = false;
    let mut stuck = 0usize;
    for _ in 0..BUDGET {
        let before = x.clone();
        match solver.step(&obj, &mut x) {
            Ok(rep) if rep.grad_norm < FMAX => {
                converged = true;
                break;
            }
            Ok(_) => {
                let same = x.iter().zip(before.iter()).all(|(a, b)| a == b);
                if same {
                    stuck += 1;
                    if stuck == 3 {
                        break;
                    }
                } else {
                    stuck = 0;
                }
            }
            Err(_) => break,
        }
    }
    let (energy, g) = fg(x.view());
    (
        calls.load(Ordering::Relaxed),
        energy,
        fmax_of(&g),
        converged,
    )
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn bench<F>(problem: &str, x0: &Array1<f64>, fg: F) -> Vec<Outcome>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync + Copy,
{
    let (e0, g0) = fg(x0.view());
    println!("init\t{problem}\t{:.16e}\t{:.16e}", e0, fmax_of(&g0));
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut rows = Vec::new();
    for arm in ARMS {
        let mut times = Vec::with_capacity(REPEATS);
        let mut first = None;
        for rep in 0..REPEATS {
            let t0 = Instant::now();
            let got = once(arm, x0, fg);
            times.push(t0.elapsed().as_secs_f64() * 1e3);
            match &first {
                None => first = Some(got),
                Some(prev) => {
                    assert_eq!(prev.0, got.0, "{problem} {} repeat {rep}", arm.name());
                    assert!((prev.1 - got.1).abs() <= 1e-9, "{problem} {}", arm.name());
                }
            }
        }
        let (evals, energy, fmax, converged) = first.unwrap();
        let row = Outcome {
            evals,
            energy,
            fmax,
            ms: median(times),
            converged,
        };
        println!(
            "result\t{problem}\t{}\t{}\t{:.16e}\t{:.16e}\t{:.6}\t{}",
            arm.name(),
            row.evals,
            row.energy,
            row.fmax,
            row.ms,
            u8::from(row.converged)
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
        rows.push(row);
    }
    rows
}

fn main() {
    println!("meta\tfmax\t{FMAX}");
    println!("meta\tmaxstep\t{MAXSTEP}");
    println!("meta\trepeats\t{REPEATS}");
    println!("meta\tbudget\t{BUDGET}");
    println!("meta\tmemory\t8");
    println!("meta\tfire_dt\t{FIRE_DT}");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    for atoms in [7usize, 13, 38] {
        let x0 = grid_start(atoms);
        bench(&format!("LJ{atoms}"), &x0, lennard_jones);
    }
    let (x0, cell) = read_con("morse_pos_0000.con");
    bench("MorsePt256", &x0, move |x: ArrayView1<f64>| morse(cell, x));
}
