// Copyright (c) rgmin developers

//! Force calls and wall time of the eOn L-BFGS step, the scaled step,
//! the Wolfe search, the geometry step, FIRE and FIRE 2.0.
//!
//! Clusters stop at a max-atom force of 0.01. Rosenbrock stops at an
//! infinity-norm of 1e-5, which is SciPy's CG default. The step cap is
//! 0.2. Repeats are serial. The median wall time is the optimizer and
//! the potential together. Evaluations do not depend on the repeat.
//!
//! `env -i PATH HOME LANG cargo run --release --example bench_suite -- bench/raw/rgmin.csv`

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rgmin::{
    Accept, Control, EonCg, FireKind, LbfgsStep, Method, Oracle, Solver, StopNorm, minimize_eon_cg,
};

const REPEATS: usize = 5;
const BUDGET: usize = 20_000;
const CAP: f64 = 0.2;
const SPACING: f64 = 1.1;
const JITTER: f64 = 0.03;

#[derive(Clone, Copy)]
enum Arm {
    Eon8,
    Eon20,
    Scaled,
    Wolfe,
    Geometry,
    Fire,
    Fire2,
    Cg,
}

fn rosenbrock(x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
    let x = x.as_slice().expect("rosenbrock x");
    let mut f = 0.0;
    g.fill(0.0);
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
        g[i] += -2.0 * a - 400.0 * x[i] * b;
        g[i + 1] += 200.0 * b;
    }
    f
}

fn lennard_jones(x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
    let n = x.len() / 3;
    let mut e = 0.0;
    g.fill(0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: [f64; 3] = std::array::from_fn(|a| x[3 * i + a] - x[3 * j + a]);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let s6 = r2.powf(-3.0);
            e += 4.0 * (s6 * s6 - s6);
            let du_over_r = 24.0 * (s6 - 2.0 * s6 * s6) / r2;
            for a in 0..3 {
                g[3 * i + a] += du_over_r * d[a];
                g[3 * j + a] -= du_over_r * d[a];
            }
        }
    }
    e
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

fn morse(cell: [f64; 3], x: ArrayView1<f64>, g: &mut [f64]) -> f64 {
    let n = x.len() / 3;
    let mut e = 0.0;
    g.fill(0.0);
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
    e
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

fn stop_of(cluster: bool, g: &Array1<f64>) -> f64 {
    if cluster {
        StopNorm::MaxAtom.of(g.view())
    } else {
        StopNorm::Inf.of(g.view())
    }
}

type Force = Box<dyn Fn(ArrayView1<f64>, &mut [f64]) -> f64 + Sync>;

struct Row {
    evals: usize,
    ms: f64,
    g: f64,
    f: f64,
    converged: bool,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn run_arm<F>(arm: Arm, x0: &Array1<f64>, cluster: bool, fg: &F) -> Row
where
    F: Fn(ArrayView1<f64>, &mut [f64]) -> f64 + Sync,
{
    let tol = if cluster { 0.01 } else { 1e-5 };
    let mut times = Vec::with_capacity(REPEATS);
    let mut row = Row {
        evals: 0,
        ms: 0.0,
        g: 0.0,
        f: 0.0,
        converged: false,
    };
    for rep in 0..REPEATS {
        let calls = AtomicUsize::new(0);
        let counted = |x: ArrayView1<f64>, g: &mut Array1<f64>| {
            calls.fetch_add(1, Ordering::Relaxed);
            fg(x, g.as_slice_mut().unwrap())
        };
        let obj = Oracle::unbounded(x0.len(), |x: ArrayView1<f64>| {
            let mut g = Array1::zeros(x.len());
            let f = {
                calls.fetch_add(1, Ordering::Relaxed);
                fg(x, g.as_slice_mut().unwrap())
            };
            (f, g)
        });
        let (method, istep) = match arm {
            Arm::Eon8 | Arm::Scaled | Arm::Wolfe | Arm::Geometry => {
                (Method::Lbfgs { memory: 8 }, 1.0)
            }
            Arm::Eon20 => (Method::Lbfgs { memory: 20 }, 1.0),
            Arm::Fire => (Method::Fire { kind: FireKind::V1 }, 1.0),
            Arm::Fire2 => (Method::Fire { kind: FireKind::V2 }, 0.1),
            Arm::Cg => (Method::Lbfgs { memory: 8 }, 1.0),
        };
        let control = Control {
            maxiter: BUDGET,
            gtol: tol,
            istep,
            stop_norm: if cluster {
                StopNorm::MaxAtom
            } else {
                StopNorm::Inf
            },
            ..Control::default()
        };
        let mut solver = Solver::new(method, control, x0.len());
        match arm {
            Arm::Eon8 | Arm::Eon20 => {
                solver
                    .set_lbfgs_step(LbfgsStep::Capped { max_atom: CAP })
                    .unwrap();
            }
            Arm::Scaled => {
                solver
                    .set_lbfgs_step(LbfgsStep::Scaled { max_atom: CAP })
                    .unwrap();
            }
            Arm::Wolfe => {
                solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
                solver.set_accept(Accept::Energy);
            }
            Arm::Geometry => {
                solver.set_lbfgs_step(LbfgsStep::Wolfe).unwrap();
                solver.set_atom_maxmove(CAP);
            }
            Arm::Fire => solver.set_atom_maxmove(CAP),
            Arm::Fire2 => solver.set_maxmove(CAP),
            Arm::Cg => {}
        }
        let mut x = x0.clone();
        let t0 = Instant::now();
        let converged = if matches!(arm, Arm::Cg) {
            let mut raw = x.to_vec();
            let opts = EonCg {
                max_move: CAP,
                gtol: tol,
                max_iter: BUDGET,
                cluster,
                atom_only: false,
            };
            let rep = minimize_eon_cg(
                &mut raw,
                |xs, gs| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    fg(ArrayView1::from(xs), gs)
                },
                &opts,
            );
            x = Array1::from(raw);
            rep.converged
        } else if matches!(arm, Arm::Eon8 | Arm::Eon20 | Arm::Fire | Arm::Fire2) {
            match solver.run_fill(&mut x, counted) {
                Ok(rep) => rep.grad_norm < tol,
                Err(_) => false,
            }
        } else {
            match solver.run(&obj, &mut x) {
                Ok(rep) => rep.grad_norm < tol,
                Err(_) => false,
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        if rep + 1 == REPEATS {
            let mut g = vec![0.0; x.len()];
            let f = fg(x.view(), &mut g);
            let g = Array1::from(g);
            row = Row {
                evals: calls.load(Ordering::Relaxed),
                ms: 0.0,
                g: stop_of(cluster, &g),
                f,
                converged,
            };
        }
    }
    row.ms = median(times);
    row
}

fn arm_name(arm: Arm) -> &'static str {
    match arm {
        Arm::Eon8 => "rgmin-eon",
        Arm::Eon20 => "rgmin-eon-m20",
        Arm::Scaled => "rgmin-scaled",
        Arm::Wolfe => "rgmin-wolfe",
        Arm::Geometry => "rgmin-geometry",
        Arm::Fire => "rgmin-fire",
        Arm::Fire2 => "rgmin-fire2",
        Arm::Cg => "rgmin-cg",
    }
}

fn main() {
    let path = std::env::args().nth(1);
    let mut out: Box<dyn std::io::Write> = if let Some(path) = &path {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        Box::new(std::fs::File::create(path).expect("csv"))
    } else {
        Box::new(std::io::stdout())
    };
    writeln!(
        out,
        "problem,arm,library,evals,energy,grad,median_ms,converged"
    )
    .unwrap();
    let problems: Vec<(String, Array1<f64>, bool, Force)> = {
        let (morse_x, cell) = read_con("morse_pos_0000.con");
        vec![
            (
                "Rosenbrock2".into(),
                rosenbrock_start(2),
                false,
                Box::new(rosenbrock),
            ),
            (
                "Rosenbrock10".into(),
                rosenbrock_start(10),
                false,
                Box::new(rosenbrock),
            ),
            ("LJ7".into(), grid_start(7), true, Box::new(lennard_jones)),
            ("LJ13".into(), grid_start(13), true, Box::new(lennard_jones)),
            ("LJ38".into(), grid_start(38), true, Box::new(lennard_jones)),
            (
                "MorsePt256".into(),
                morse_x,
                true,
                Box::new(move |x, g| morse(cell, x, g)),
            ),
        ]
    };
    let arms = [
        Arm::Eon8,
        Arm::Eon20,
        Arm::Scaled,
        Arm::Wolfe,
        Arm::Geometry,
        Arm::Fire,
        Arm::Fire2,
        Arm::Cg,
    ];
    for (name, x0, cluster, fg) in &problems {
        for arm in arms {
            let row = run_arm(arm, x0, *cluster, fg);
            let line = format!(
                "{name},{},rgmin,{},{:.8e},{:.8e},{:.6},{}",
                arm_name(arm),
                row.evals,
                row.f,
                row.g,
                row.ms,
                if row.converged { "yes" } else { "no" }
            );
            writeln!(out, "{line}").unwrap();
            println!("{line}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
}
