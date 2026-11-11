// Copyright (c) rgmin developers

//! Force calls and wall time of capped L-BFGS against the develop defaults.
//!
//! Default `Lbfgs` (memory 8, strong Wolfe, `norminf(g) < 1e-6`) against
//! the same solver with [`LbfgsStep::Capped`] and cap 0.2, on the extended
//! Rosenbrock function and on Lennard-Jones clusters from a jittered cubic
//! grid. A Euclidean session with memory 20 compares the geometry step
//! (`set_atom_maxmove(0.2)`) with the capped step on the same starts, and
//! on one periodic Morse cluster, stopping at max-atom force 0.01.
//! Each arm runs serially seven times. The table is the median wall time.
//! Evaluations do not depend on the repeat.
//!
//! `env -i PATH HOME LANG cargo run --release --example lbfgs_capped`

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, Lbfgs, LbfgsStep, Method, Oracle, Solver, StopNorm};

const REPEATS: usize = 7;
const BUDGET: usize = 100_000;
const CAP: f64 = 0.2;
const SPACING: f64 = 1.1;
const JITTER: f64 = 0.03;

fn rosenbrock(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
        g[i] += -2.0 * a - 400.0 * x[i] * b;
        g[i + 1] += 200.0 * b;
    }
    (f, g)
}

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

fn rosenbrock_start(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }))
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

fn ginf(g: &Array1<f64>) -> f64 {
    g.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn max_atom(g: &Array1<f64>) -> f64 {
    StopNorm::MaxAtom.of(g.view())
}

type Force = dyn Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync;

struct Row {
    evals: usize,
    ms: f64,
    g: f64,
    f: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time_minimize(
    x0: &Array1<f64>,
    capped: bool,
    fg: fn(ArrayView1<f64>) -> (f64, Array1<f64>),
) -> Row {
    let mut times = Vec::with_capacity(REPEATS);
    let mut row = Row {
        evals: 0,
        ms: 0.0,
        g: 0.0,
        f: 0.0,
    };
    for rep in 0..REPEATS {
        let mut opt = if capped {
            Lbfgs::capped(CAP)
        } else {
            Lbfgs::default()
        };
        let t0 = Instant::now();
        let (f, x, evals) = opt.minimize(x0.view(), BUDGET, |v| Some(fg(v)));
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        if rep + 1 == REPEATS {
            let g = fg(x.view()).1;
            row = Row {
                evals,
                ms: 0.0,
                g: ginf(&g),
                f,
            };
        }
    }
    row.ms = median(times);
    row
}

fn time_session(x0: &Array1<f64>, capped: bool, tol: f64, fg: &Force) -> Row {
    let mut times = Vec::with_capacity(REPEATS);
    let mut row = Row {
        evals: 0,
        ms: 0.0,
        g: 0.0,
        f: 0.0,
    };
    for rep in 0..REPEATS {
        let calls = AtomicUsize::new(0);
        let obj = Oracle::unbounded(x0.len(), |x: ArrayView1<f64>| {
            calls.fetch_add(1, Ordering::Relaxed);
            fg(x)
        });
        let control = Control {
            maxiter: BUDGET,
            gtol: tol,
            istep: CAP,
            stop_norm: StopNorm::MaxAtom,
            ..Control::default()
        };
        let mut solver = Solver::new(Method::Lbfgs { memory: 20 }, control, x0.len());
        if capped {
            solver
                .set_lbfgs_step(LbfgsStep::Capped { max_atom: CAP })
                .unwrap();
        } else {
            solver.set_atom_maxmove(CAP);
        }
        let mut x = x0.clone();
        let t0 = Instant::now();
        for _ in 0..BUDGET {
            match solver.step(&obj, &mut x) {
                Ok(rep) if rep.grad_norm < tol => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        if rep + 1 == REPEATS {
            let (f, g) = fg(x.view());
            row = Row {
                evals: calls.load(Ordering::Relaxed),
                ms: 0.0,
                g: max_atom(&g),
                f,
            };
        }
    }
    row.ms = median(times);
    row
}

fn print_row(problem: &str, arm: &str, row: &Row) {
    println!(
        "| {problem} | {arm} | {:>6} | {:>10.3} | {:>9.2e} | {:>14.6e} |",
        row.evals, row.ms, row.g, row.f
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

fn main() {
    println!("| problem | arm | evals | median ms | final g | f |");
    println!("|---|---|---:|---:|---:|---:|");
    for (label, x0, fg) in [
        (
            "Rosenbrock 2",
            rosenbrock_start(2),
            rosenbrock as fn(ArrayView1<f64>) -> (f64, Array1<f64>),
        ),
        ("Rosenbrock 10", rosenbrock_start(10), rosenbrock),
        ("Rosenbrock 100", rosenbrock_start(100), rosenbrock),
        ("LJ7", grid_start(7), lennard_jones),
        ("LJ13", grid_start(13), lennard_jones),
        ("LJ38", grid_start(38), lennard_jones),
    ] {
        print_row(label, "Wolfe", &time_minimize(&x0, false, fg));
        print_row(label, "capped 0.2", &time_minimize(&x0, true, fg));
        print_row(
            label,
            "geometry session",
            &time_session(&x0, false, 0.01, &fg),
        );
        print_row(label, "capped session", &time_session(&x0, true, 0.01, &fg));
    }
    let (x0, cell) = read_con("morse_pos_0000.con");
    let fg = move |x: ArrayView1<f64>| morse(cell, x);
    print_row(
        "Morse Pt256",
        "geometry session",
        &time_session(&x0, false, 0.01, &fg),
    );
    print_row(
        "Morse Pt256",
        "capped session",
        &time_session(&x0, true, 0.01, &fg),
    );
}
