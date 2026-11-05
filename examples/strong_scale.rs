//! CPU strong scaling: one fixed system, thread counts 1, 2, 4, ... up
//! to the physical cores.
//!
//! The system is a perturbed periodic FCC crystal of `4 L^3` atoms with
//! one Morse bond (Pt parameters of `relax_morse`) to each of the 12
//! nearest neighbours of the ideal lattice. The lattice constant puts
//! every bond at the Morse minimum, so the ideal lattice is the minimum
//! at `E = -6 N D0`. The bond list is fixed, which keeps the potential
//! analytic, and is valid while displacements stay far below the
//! nearest-neighbour distance.
//!
//! Two methods run on each rayon pool: L-BFGS (m = 10, strong Wolfe) and
//! FIRE 2.0 with the Guenole 2020 preset. The force evaluation and the
//! `par` vector seam both run on the pool. Every run starts from the same
//! geometry and stops at a max per-atom force below 0.01 eV/Angstrom.
//!
//! CSV goes to stdout. `wall_s` is the median over the repeats,
//! `max_dev` the largest coordinate difference from the first 1-thread
//! run over all repeats, `efficiency` the speedup over 1 thread divided
//! by the thread count. The harness is CPU only; the `gpu` column stays
//! empty. It exits non-zero when a run does not converge.
//!
//! `cargo run --release --features par --example strong_scale -- \
//!   --cells 20 --repeats 5 [--max-threads T] [--hardware NAME]`

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use rgmin::{Control, FireKind, FirePreset, LineSearch, Method, Oracle, Solver, StopNorm};

/// Pt Morse parameters of tests/newton_cg_morse.rs: eV, 1/Angstrom, Angstrom.
const D0: f64 = 0.7102;
const ALPHA: f64 = 1.6047;
const R0: f64 = 2.897;

/// Uniform per-coordinate displacement bound, Angstrom: 3.5 % of `R0`.
const AMP: f64 = 0.1;
const SEED: u64 = 0;

const FMAX: f64 = 0.01;
const BUDGET: usize = 2000;

/// Half-lattice offsets of the 12 FCC nearest neighbours.
const NN: [[i64; 3]; 12] = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, 1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, 1, -1],
    [0, -1, 1],
    [0, -1, -1],
];

struct System {
    x0: Array1<f64>,
    nbr: Vec<[usize; 12]>,
    box_len: f64,
}

/// Perturbed FCC positions and the nearest-neighbour list of the ideal
/// lattice. Atoms sit on the half-lattice points whose coordinates sum
/// to an even number.
fn fcc(cells: usize) -> System {
    let a = R0 * std::f64::consts::SQRT_2;
    let side = 2 * cells as i64;
    let mut index = vec![usize::MAX; (side * side * side) as usize];
    let mut sites = Vec::new();
    for h in 0..side * side * side {
        let p = [h / (side * side), (h / side) % side, h % side];
        if (p[0] + p[1] + p[2]) % 2 == 0 {
            index[h as usize] = sites.len();
            sites.push(p);
        }
    }
    let flat = |p: [i64; 3]| -> usize {
        let w = |v: i64| v.rem_euclid(side);
        index[(w(p[0]) * side * side + w(p[1]) * side + w(p[2])) as usize]
    };
    let nbr = sites
        .iter()
        .map(|p| NN.map(|d| flat([p[0] + d[0], p[1] + d[1], p[2] + d[2]])))
        .collect();
    let mut rng = StdRng::seed_from_u64(SEED);
    let x0 = sites
        .iter()
        .flat_map(|p| p.map(|v| v as f64 * a / 2.0))
        .map(|v| v + rng.random_range(-AMP..AMP))
        .collect();
    System {
        x0,
        nbr,
        box_len: cells as f64 * a,
    }
}

/// Morse energy and gradient over the fixed bonds, one atom per task.
fn morse(sys: &System, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let x = x.as_slice().expect("contiguous");
    let mut g = vec![0.0; x.len()];
    let e: f64 = g
        .par_chunks_mut(3)
        .enumerate()
        .map(|(i, gi)| {
            let mut ei = 0.0;
            for &j in &sys.nbr[i] {
                let mut d = [0.0; 3];
                for k in 0..3 {
                    let dx = x[3 * i + k] - x[3 * j + k];
                    d[k] = dx - sys.box_len * (dx / sys.box_len).round();
                }
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let ex = (-ALPHA * (r - R0)).exp();
                ei += 0.5 * D0 * (ex * ex - 2.0 * ex);
                let du = 2.0 * D0 * ALPHA * ex * (1.0 - ex);
                for k in 0..3 {
                    gi[k] += du * d[k] / r;
                }
            }
            ei
        })
        .sum();
    (e, Array1::from(g))
}

struct Run {
    wall: f64,
    calls: usize,
    steps: usize,
    fmax: f64,
    x: Array1<f64>,
}

fn relax(sys: &System, name: &str) -> Run {
    let n = sys.x0.len();
    let (method, istep) = match name {
        "lbfgs" => (Method::Lbfgs { memory: 10 }, 0.2),
        _ => (Method::Fire { kind: FireKind::V2 }, 0.1),
    };
    let control = Control {
        maxiter: BUDGET,
        gtol: FMAX,
        istep,
        maxmove: None,
    };
    let calls = AtomicUsize::new(0);
    let obj = Oracle::unbounded(n, |x: ArrayView1<f64>| {
        calls.fetch_add(1, Ordering::Relaxed);
        morse(sys, x)
    });
    let start = Instant::now();
    let mut solver = Solver::new(method, control, n);
    solver.set_atom_maxmove(0.2);
    solver.set_stop_norm(StopNorm::MaxAtom);
    match name {
        "lbfgs" => solver.set_linesearch(LineSearch::Wolfe {
            c1: 1e-4,
            c2: 0.9,
            maxiter: 40,
        }),
        _ => solver
            .set_fire_preset(FirePreset::Guenole2020)
            .expect("fire2 preset"),
    }
    let mut x = sys.x0.clone();
    let (mut steps, mut fmax) = (0, f64::INFINITY);
    while steps < BUDGET && fmax >= FMAX {
        fmax = solver.step(&obj, &mut x).expect("step").grad_norm;
        steps += 1;
    }
    Run {
        wall: start.elapsed().as_secs_f64(),
        calls: calls.load(Ordering::Relaxed),
        steps,
        fmax,
        x,
    }
}

/// CPU model name and physical core count from /proc/cpuinfo, falling
/// back to the logical count.
fn host() -> (String, usize) {
    let info = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |l: &str| l.split_once(':').map(|(_, v)| v.trim().to_string());
    let model = info
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(field)
        .unwrap_or_else(|| "unknown".into());
    let mut cores = std::collections::BTreeSet::new();
    let mut socket = String::new();
    for l in info.lines() {
        if l.starts_with("physical id") {
            socket = field(l).unwrap_or_default();
        } else if l.starts_with("core id") {
            cores.insert((socket.clone(), field(l).unwrap_or_default()));
        }
    }
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    let physical = if cores.is_empty() {
        logical
    } else {
        cores.len()
    };
    (model, physical)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |key: &str| {
        args.iter()
            .position(|a| a == key)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let num = |key: &str, default: usize| arg(key).map_or(default, |v| v.parse().expect(key));
    let (model, physical) = host();
    let hardware = arg("--hardware").unwrap_or(model);
    let max_threads = num("--max-threads", physical).max(1);
    let repeats = num("--repeats", 5).max(1);
    let sys = fcc(num("--cells", 20).max(2));
    let mut threads: Vec<usize> = (0..)
        .map(|k| 1 << k)
        .take_while(|&t| t < max_threads)
        .collect();
    threads.push(max_threads);

    println!(
        "hardware,cores,gpu,method,threads,n,repeats,wall_s,force_calls,iterations,grad_norm,max_dev,speedup,efficiency"
    );
    let mut ok = true;
    for name in ["lbfgs", "fire2"] {
        let mut base: Option<(f64, Array1<f64>)> = None;
        for &t in &threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("thread pool");
            let runs: Vec<Run> = (0..repeats)
                .map(|_| pool.install(|| relax(&sys, name)))
                .collect();
            let mut walls: Vec<f64> = runs.iter().map(|r| r.wall).collect();
            walls.sort_by(f64::total_cmp);
            let m = walls.len();
            let wall = (walls[(m - 1) / 2] + walls[m / 2]) / 2.0;
            let (t1, x1) = base.get_or_insert_with(|| (wall, runs[0].x.clone()));
            let dev = runs
                .iter()
                .flat_map(|r| (&r.x - &*x1).mapv(f64::abs))
                .fold(0.0, f64::max);
            let r = &runs[0];
            ok &= runs.iter().all(|r| r.fmax < FMAX);
            let speedup = *t1 / wall;
            println!(
                "\"{hardware}\",{physical},,{name},{t},{},{repeats},{wall:.6},{},{},{:.6e},{dev:.3e},{speedup:.3},{:.3}",
                sys.x0.len(),
                r.calls,
                r.steps,
                r.fmax,
                speedup / t as f64,
            );
        }
    }
    if !ok {
        eprintln!("strong_scale: a run did not reach fmax < {FMAX} eV/Angstrom");
        std::process::exit(1);
    }
}
