# Copyright (c) rgmin developers
"""ASE Optimizer drop-in on bound Lennard-Jones clusters and a Cu cell.

The previous cubic lattice at spacing 1.5 is not this script. On that
start ASE LBFGS with alpha=70 walks the atoms apart: one LJ13 seed
goes from radius 2.51 to 6.48 while the maximum force falls through
0.01, because the pairs have gone. The energy is then about -0.02.
A compact start with the same alpha stays bound, so the lattice was
the cause, not the initial Hessian by itself.

LJ13 starts from the Mackay icosahedron and LJ38 from the Oh
truncated octahedron, the Cambridge Cluster Database global minima
(https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/13 and
.../38; energies -44.326801 and -173.928427). Each of ten seeds adds
Normal(0, 0.05) from PCG64(20261113 + seed). A second set draws
points uniformly in a sphere of density 1 with pair distances at
least 0.9. Cu64 is the fcc (4, 4, 4) cell rattled the same way.

A run counts as converged only when the maximum force is below 0.01
and the largest distance from the centroid stays under twice the
value at the start. Anything past that radius is a dissociation.
ASE LBFGS is run twice: alpha=70, the ASE default, and alpha=10.

rgmin FIRE is Bitzek's velocity-Verlet step, with the force mixed
after the move. The parameters are ASE's defaults (dt=0.1, dtmax=1,
Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99) and the longest atom
moves at most 0.2. ASE FIRE mixes before the Euler kick and caps the
Euclidean length of the whole displacement.

    python benches/ase_dropin.py
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.optimize import FIRE as AseFire
from ase.optimize import LBFGS as AseLBFGS

from rgmin.optimize import FIRE as RgminFire
from rgmin.optimize import LBFGS as RgminLBFGS

# Cambridge Cluster Database, Doye and Wales. Units of sigma.
# https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/13
LJ13 = np.array(
    [
        [1.0132226417, 0.3329955686, 0.1812866397],
        [0.7255989775, -0.7660449415, 0.2388625373],
        [0.7293356067, -0.2309436666, -0.7649239428],
        [0.3513618941, 0.8291166557, -0.5995702064],
        [0.3453146118, -0.0366957540, 1.0245903005],
        [0.1140240770, 0.9491685999, 0.5064104273],
        [-1.0132240213, -0.3329960305, -0.1812867552],
        [-0.1140234764, -0.9491689127, -0.5064103454],
        [-0.3513615244, -0.8291170821, 0.5995701458],
        [-0.3453152548, 0.0366956843, -1.0245902691],
        [-0.7255983925, 0.7660457628, -0.2388624662],
        [-0.7293359733, 0.2309438428, 0.7649237858],
        [0.0000008339, 0.0000002733, 0.0000001488],
    ]
)
# https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/38
LJ38 = np.array(
    [
        [0.1947679907, 0.3306365642, 1.7069272101],
        [1.1592174250, -1.1514615100, -0.6254746298],
        [1.4851406793, -0.0676273830, 0.9223060046],
        [-0.1498046416, 1.4425168343, -0.9785553065],
        [1.4277261305, 0.3530265376, -0.9475378022],
        [-0.6881246261, -1.5737014419, -0.3328844168],
        [-1.4277352637, -0.3530034531, 0.9475270683],
        [0.6881257085, 1.5736904826, 0.3329032458],
        [-1.1592204530, 1.1514535263, 0.6254777879],
        [0.1498035273, -1.4424985165, 0.9785685322],
        [-1.4851196066, 0.0676193562, -0.9223231092],
        [-0.7057028384, 0.6207073550, -1.4756523155],
        [-0.8745359533, 0.4648140463, 1.4422103492],
        [-0.9742077067, -0.8837261792, -1.1536019836],
        [-0.1947765396, -0.3306358487, -1.7069179299],
        [0.3759933035, -1.7072373106, -0.0694439840],
        [-1.7124296000, 0.3336352522, 0.1307959669],
        [0.9143159284, 1.3089975397, -0.7151210582],
        [-0.3759920260, 1.7072300336, 0.0694634263],
        [1.7124281219, -0.3336312342, -0.1308207313],
        [-0.9143187026, -1.3089785474, 0.7151290509],
        [0.9742085109, 0.8837023041, 1.1536069633],
        [0.7057104439, -0.6206907639, 1.4756502961],
        [0.8745319670, -0.4648127187, -1.4422106957],
        [-1.1954804901, -0.6171923123, -0.1021449363],
        [0.0917363053, -1.0144887859, -0.8848410405],
        [0.9276243144, -0.8836123311, 0.4234140820],
        [1.1954744473, 0.6171883800, 0.1021399054],
        [-0.9276176774, 0.8836123556, -0.4234173533],
        [-0.3595942315, -0.4863167551, 1.2061133825],
        [0.3595891589, 0.4863295901, -1.2061152849],
        [-0.0917352078, 1.0144694592, 0.8848400639],
        [0.6410702480, -0.1978633363, -0.3898095439],
        [-0.4162942817, -0.0651798741, -0.6515502084],
        [0.1334019604, 0.7474406294, -0.1600033264],
        [-0.6410732823, 0.1978593218, 0.3898012337],
        [0.4162968444, 0.0651733322, 0.6515490914],
        [-0.1333998872, -0.7474445984, 0.1600019961],
    ]
)

DEFAULT_OUT = Path(__file__).resolve().parent / "ase_dropin" / "runs.csv"
FMAX = 0.01
STEPS = 5000
SEEDS = 10
MASTER = 20261113
PERTURB = 0.05
MIN_PAIR = 0.9
# ASE's default initial inverse Hessian is I/70. The reduced-unit row uses 10.
ASE_LBFGS_ALPHA = 70.0
ASE_LBFGS_REDUCED_ALPHA = 10.0

SYSTEMS = ("LJ13", "LJ38", "LJ13-sphere", "LJ38-sphere", "Cu64")
ARMS = (
    "rgmin-lbfgs",
    "rgmin-fire",
    "ase-lbfgs",
    "ase-lbfgs-a10",
    "ase-fire",
    "ase-bfgs",
)


class Counted(Calculator):
    """One increment per geometry the inner calculator evaluates."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, inner: Calculator) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.calls += 1
        self.inner.calculate(atoms, properties, system_changes)
        self.results = dict(self.inner.results)


def radius(positions: np.ndarray) -> float:
    center = positions.mean(axis=0)
    return float(np.linalg.norm(positions - center, axis=1).max())


def min_pair(positions: np.ndarray) -> float:
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.sqrt((delta * delta).sum(axis=-1))
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def lj_calc() -> Counted:
    return Counted(LennardJones(epsilon=1.0, sigma=1.0, rc=10.0))


def rattled(positions: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(MASTER + seed))
    return positions + rng.normal(0.0, PERTURB, size=positions.shape)


def sphere(n: int, seed: int) -> np.ndarray:
    """Uniform points in a ball of density 1, no pair closer than 0.9."""
    volume = float(n)
    ball = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    rng = np.random.Generator(np.random.PCG64(MASTER + 1000 * n + seed))
    points: list[np.ndarray] = []
    for _ in range(200_000):
        if len(points) == n:
            break
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        trial = direction * ball * rng.random() ** (1.0 / 3.0)
        if all(np.linalg.norm(trial - other) >= MIN_PAIR for other in points):
            points.append(trial)
    if len(points) != n:
        raise RuntimeError(f"could not place {n} points at density 1")
    return np.asarray(points)


def make_positions(system: str, seed: int) -> tuple[np.ndarray, str]:
    if system == "LJ13":
        return rattled(LJ13, seed), "lj"
    if system == "LJ38":
        return rattled(LJ38, seed), "lj"
    if system == "LJ13-sphere":
        return sphere(13, seed), "lj"
    if system == "LJ38-sphere":
        return sphere(38, seed), "lj"
    atoms = bulk("Cu", "fcc", a=3.615, cubic=False)
    atoms *= (4, 4, 4)
    return rattled(atoms.get_positions(), seed), "emt"


def make_atoms(system: str, seed: int) -> Atoms:
    positions, kind = make_positions(system, seed)
    if kind == "emt":
        atoms = bulk("Cu", "fcc", a=3.615, cubic=False)
        atoms *= (4, 4, 4)
        atoms.set_positions(positions)
        atoms.calc = Counted(EMT())
        return atoms
    atoms = Atoms(f"X{len(positions)}", positions=positions, pbc=False)
    atoms.calc = lj_calc()
    return atoms


def optimize(atoms: Atoms, name: str, steps: int) -> tuple[bool, int]:
    if name == "rgmin-lbfgs":
        opt = RgminLBFGS(atoms, logfile=None, eon_cap=True, maxstep=0.2)
    elif name == "rgmin-fire":
        opt = RgminFire(atoms, logfile=None)
    elif name == "ase-lbfgs":
        opt = AseLBFGS(atoms, logfile=None, maxstep=0.2, alpha=ASE_LBFGS_ALPHA)
    elif name == "ase-lbfgs-a10":
        opt = AseLBFGS(atoms, logfile=None, maxstep=0.2, alpha=ASE_LBFGS_REDUCED_ALPHA)
    elif name == "ase-fire":
        opt = AseFire(atoms, logfile=None)
    else:
        opt = BFGS(atoms, logfile=None)
    reached = bool(opt.run(fmax=FMAX, steps=steps))
    return reached, opt.nsteps


def one(system: str, seed: int, name: str, steps: int) -> dict:
    atoms = make_atoms(system, seed)
    start = radius(atoms.get_positions())
    calc = atoms.calc
    assert isinstance(calc, Counted)
    started = time.perf_counter()
    reached, _nsteps = optimize(atoms, name, steps)
    elapsed = time.perf_counter() - started
    end = radius(atoms.get_positions())
    dissociated = end > 2.0 * start
    energy = float(atoms.get_potential_energy())
    return {
        "system": system,
        "seed": seed,
        "optimizer": name,
        "force_calls": calc.calls,
        "energy": energy,
        "seconds": elapsed,
        "fmax_reached": int(reached),
        "dissociated": int(dissociated),
        "converged": int(reached and not dissociated),
    }


def medians(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["optimizer"]), []).append(row)
    summary = []
    for (system, optimizer), group in sorted(grouped.items()):
        bound = [row for row in group if row["converged"]]
        summary.append(
            {
                "system": system,
                "optimizer": optimizer,
                "runs": len(group),
                "converged": sum(row["converged"] for row in group),
                "dissociated": sum(row["dissociated"] for row in group),
                "force_calls": float(np.median([row["force_calls"] for row in bound])) if bound else float("nan"),
                "energy": float(np.median([row["energy"] for row in bound])) if bound else float("nan"),
                "seconds": float(np.median([row["seconds"] for row in bound])) if bound else float("nan"),
            }
        )
    return summary


def print_summary(summary: list[dict]) -> None:
    header = (
        f"{'system':<12} {'optimizer':<14} {'ok':>6} {'flew':>5} "
        f"{'calls':>8} {'energy':>14} {'seconds':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        energy = f"{row['energy']:14.6f}" if np.isfinite(row["energy"]) else f"{'—':>14}"
        calls = f"{row['force_calls']:8.1f}" if np.isfinite(row["force_calls"]) else f"{'—':>8}"
        seconds = f"{row['seconds']:8.4f}" if np.isfinite(row["seconds"]) else f"{'—':>8}"
        print(
            f"{row['system']:<12} {row['optimizer']:<14} "
            f"{row['converged']:>2}/{row['runs']:<3} {row['dissociated']:>5} "
            f"{calls} {energy} {seconds}"
        )


def bare_energies() -> None:
    for name, positions, target in (
        ("LJ13", LJ13, -44.326801),
        ("LJ38", LJ38, -173.928427),
    ):
        atoms = Atoms(f"X{len(positions)}", positions=positions, pbc=False)
        atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=10.0)
        energy = float(atoms.get_potential_energy())
        print(f"unperturbed {name}: {energy:.6f} (database {target:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    bare_energies()
    rows: list[dict] = []
    for seed in range(args.seeds):
        for system in SYSTEMS:
            for name in ARMS:
                row = one(system, seed, name, args.steps)
                rows.append(row)
                print(
                    f"{system} seed {seed} {name}: calls={row['force_calls']} "
                    f"energy={row['energy']:.6f} t={row['seconds']:.3f}s "
                    f"fmax={row['fmax_reached']} dissociated={row['dissociated']}",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "seed",
                "optimizer",
                "force_calls",
                "energy",
                "seconds",
                "fmax_reached",
                "dissociated",
                "converged",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print()
    print("Medians are over runs that reached fmax without dissociating.")
    print(f"ASE LBFGS alpha={ASE_LBFGS_ALPHA:g}; ase-lbfgs-a10 uses alpha={ASE_LBFGS_REDUCED_ALPHA:g}.")
    print_summary(medians(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
