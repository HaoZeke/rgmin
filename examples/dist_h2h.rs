// Copyright (c) rgmin developers

//! Fifty-start distribution of the line-search-free default, the
//! opt-in Wolfe search, the ASE-matched cap, and FIRE 2.0.
//!
//! `benches/dist_h2h.py` runs ASE on the starts this binary writes.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rgmin::{Accept, Control, FireKind, LbfgsStep, Method, Oracle, Solver, StopNorm};

const NSTART: u64 = 50;
const BUDGET: usize = 5_000;
const FMAX: f64 = 0.01;
const ASE_H0: f64 = 1.0 / 70.0;

#[derive(Clone, Copy)]
enum Arm {
    Default,
    Matched,
    Wolfe,
    Fire2,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "rgmin-default",
            Self::Matched => "rgmin-matched",
            Self::Wolfe => "rgmin-wolfe",
            Self::Fire2 => "rgmin-fire2",
        }
    }
}

const ARMS: [Arm; 4] = [Arm::Default, Arm::Matched, Arm::Wolfe, Arm::Fire2];

fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let s6 = r2.powi(-3);
            e += 4.0 * (s6 * s6 - s6);
            let du = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for a in 0..3 {
                g[3 * i + a] += du * d[a];
                g[3 * j + a] -= du * d[a];
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

fn lj_start(n: usize, seed: u64) -> Array1<f64> {
    let side = (n as f64 / 0.5).cbrt();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x: Vec<f64> = Vec::with_capacity(3 * n);
    while x.len() < 3 * n {
        let p: [f64; 3] = std::array::from_fn(|_| rng.random::<f64>() * side);
        let clear = x.chunks(3).all(|q| {
            let r2 = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2);
            r2 >= 0.85 * 0.85
        });
        if clear {
            x.extend(p);
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

fn run<F>(arm: Arm, x0: &Array1<f64>, fg: F) -> (usize, f64, bool)
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Copy + Send + Sync,
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
            Arm::Fire2 => 0.1,
            _ => 1.0,
        },
        stop_norm: StopNorm::MaxAtom,
        ..Control::default()
    };
    let method = match arm {
        Arm::Matched => Method::Lbfgs { memory: 100 },
        Arm::Fire2 => Method::Fire { kind: FireKind::V2 },
        _ => Method::Lbfgs { memory: 8 },
    };
    let mut solver = Solver::new(method, control, x0.len());
    match arm {
        Arm::Default => {}
        Arm::Matched => solver.set_lbfgs_fixed_h0(Some(ASE_H0)).expect("h0"),
        Arm::Wolfe => {
            solver.set_lbfgs_step(LbfgsStep::Wolfe).expect("wolfe");
            solver.set_accept(Accept::Energy);
        }
        Arm::Fire2 => solver.set_maxmove(0.2),
    }
    let mut x = x0.clone();
    let mut ok = false;
    for _ in 0..BUDGET {
        match solver.step(&obj, &mut x) {
            Ok(rep) if rep.grad_norm < FMAX => {
                ok = true;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let (energy, _) = fg(x.view());
    (calls.load(Ordering::Relaxed), energy, ok)
}

fn stats(calls: &[usize], energy: &[f64], conv: &[bool]) -> (f64, usize, f64, usize, usize) {
    let n = calls.len();
    let mut c = calls.to_vec();
    c.sort_unstable();
    let q1 = c[n / 4];
    let q3 = c[3 * n / 4];
    let med = if n % 2 == 0 {
        (c[n / 2 - 1] + c[n / 2]) as f64 / 2.0
    } else {
        c[n / 2] as f64
    };
    let mut e = energy.to_vec();
    e.sort_by(f64::total_cmp);
    let em = if n % 2 == 0 {
        0.5 * (e[n / 2 - 1] + e[n / 2])
    } else {
        e[n / 2]
    };
    let k = conv.iter().filter(|c| **c).count();
    (med, q3 - q1, em, k, n)
}

fn bench<F>(problem: &str, starts: &[Array1<f64>], fg: F, starts_out: &mut dyn Write)
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Copy + Send + Sync,
{
    for (i, x0) in starts.iter().enumerate() {
        write!(starts_out, "{problem}\t{i}").unwrap();
        for v in x0.iter() {
            write!(starts_out, "\t{v:.16e}").unwrap();
        }
        writeln!(starts_out).unwrap();
    }
    for arm in ARMS {
        let mut calls = Vec::new();
        let mut energy = Vec::new();
        let mut conv = Vec::new();
        for x0 in starts {
            let (c, e, ok) = run(arm, x0, fg);
            calls.push(c);
            energy.push(e);
            conv.push(ok);
        }
        let (med, iqr, em, k, n) = stats(&calls, &energy, &conv);
        println!(
            "summary\t{problem}\t{}\t{n}\t{k}\t{med:.1}\t{iqr}\t{em:.6}",
            arm.name()
        );
        let _ = std::io::stdout().flush();
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("starts path");
    let mut starts_out = std::fs::File::create(&path).expect("starts file");
    println!("meta\tfmax\t{FMAX}");
    println!("meta\tbudget\t{BUDGET}");
    println!("meta\tcap\t{}", rgmin::EON_MAX_MOVE);
    for n in [13usize, 38, 75] {
        let starts: Vec<_> = (0..NSTART).map(|s| lj_start(n, s)).collect();
        bench(&format!("LJ{n}"), &starts, lj, &mut starts_out);
    }
    let (x0, cell) = read_con("morse_pos_0000.con");
    bench("MorsePt256", &[x0], move |x| morse(cell, x), &mut starts_out);
}
