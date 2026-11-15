# Copyright (c) rgmin developers
"""``rgmin.optimize`` against ``ase.optimize`` on ASE-sized systems.

Each optimizer pair is the same class name imported from the two
packages, constructed with nothing but ``logfile`` so both sides run
at their defaults: BFGS, LBFGS, LBFGSLineSearch, FIRE and FIRE2. rgmin
has no ODE12r, so ASE's ODE12r is run as a reference row only.

The systems are built with ASE and seeded, so every run starts from
the same geometry:

- Cu55: the three-shell Cu icosahedron, EMT, rattled 0.05.
- Pt38: the Pt truncated octahedron, EMT, rattled 0.05.
- Cu(111)+O and Pt(111)+O: a 3x3x4 slab with O in the fcc hollow,
  EMT, the two bottom layers fixed with FixAtoms, rattled 0.05.
- LJ38: the Cambridge Cluster Database Oh minimum (see
  ``ase_dropin.py``), ASE LennardJones with sigma = epsilon = 1 and
  rc = 10, rattled 0.05.
- Cu107 vacancy cell: the cubic 3x3x3 Cu supercell less one atom,
  rattled 0.05, cell scaled by 1.02, relaxed in a FrechetCellFilter.

Every run stops at fmax 0.01 (eV/A, or reduced units for LJ38) or
after the step budget. Force calls are distinct geometries reaching
the calculator; steps is ``opt.nsteps``. Both are checked to be the
same on every repeat. Wall time is the whole ``run`` call, calculator
included; EMT and LennardJones are Python, so it is mostly the
calculator.

    python benches/ase_optimize.py [--repeats 5] [--steps 2000]

The table is benches/ase_optimize/results.md. Every repeat is logged
to benches/ase_optimize/raw/, which is not committed.
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import ase
import ase.optimize as ase_optimize
import numpy as np
from ase.build import add_adsorbate, bulk, fcc111
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.cluster import Icosahedron, Octahedron
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter

from rgmin import optimize as rgmin_optimize
from rgmin._core import version as rgmin_version

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase_dropin import LJ38  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "ase_optimize"
FMAX = 0.01
PAIRS = ("BFGS", "LBFGS", "LBFGSLineSearch", "FIRE", "FIRE2")
REFERENCE = ("ODE12r",)


class Counted(Calculator):
    """Count the geometries the inner calculator evaluates."""

    def __init__(self, inner: Calculator) -> None:
        super().__init__()
        self.inner = inner
        self.implemented_properties = list(inner.implemented_properties)
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.calls += 1
        self.inner.calculate(atoms, properties, system_changes)
        self.results = dict(self.inner.results)


def _cluster(atoms, seed):
    atoms.rattle(stdev=0.05, seed=seed)
    atoms.calc = Counted(EMT())
    return atoms, atoms


def _slab(metal, seed):
    slab = fcc111(metal, (3, 3, 4), vacuum=8.0)
    add_adsorbate(slab, "O", 1.2, "fcc")
    slab.rattle(stdev=0.05, seed=seed)
    slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.tag >= 3]))
    slab.calc = Counted(EMT())
    return slab, slab


def _lj38():
    atoms = ase.Atoms("Ar38", positions=LJ38)
    atoms.rattle(stdev=0.05, seed=5)
    atoms.calc = Counted(LennardJones(sigma=1.0, epsilon=1.0, rc=10.0))
    return atoms, atoms


def _vacancy_cell():
    atoms = bulk("Cu", cubic=True) * (3, 3, 3)
    del atoms[0]
    atoms.rattle(stdev=0.05, seed=6)
    atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)
    atoms.calc = Counted(EMT())
    return FrechetCellFilter(atoms), atoms


CASES = {
    "Cu55": lambda: _cluster(Icosahedron("Cu", 3), 1),
    "Pt38": lambda: _cluster(Octahedron("Pt", 4, cutoff=1), 2),
    "Cu(111)+O": lambda: _slab("Cu", 3),
    "Pt(111)+O": lambda: _slab("Pt", 4),
    "LJ38": _lj38,
    "Cu107-vac-cell": _vacancy_cell,
}


def _fmax(system) -> float:
    forces = np.asarray(system.get_forces()).reshape(-1, 3)
    return float(np.linalg.norm(forces, axis=1).max())


def run_once(case: str, library: str, name: str, steps: int) -> dict:
    system, atoms = CASES[case]()
    module = ase_optimize if library == "ase" else rgmin_optimize
    opt = getattr(module, name)(system, logfile=io.StringIO())
    atoms.calc.calls = 0
    start = time.perf_counter()
    try:
        converged = bool(opt.run(fmax=FMAX, steps=steps))
        error = ""
    except Exception as exc:  # recorded in the table, never hidden
        converged = False
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - start
    return {
        "case": case,
        "library": library,
        "optimizer": name,
        "steps": int(getattr(opt, "nsteps", -1)),
        "calls": atoms.calc.calls,
        "energy": float(atoms.get_potential_energy()),
        "fmax": _fmax(system),
        "converged": converged,
        "wall": wall,
        "error": error,
    }


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment() -> list[str]:
    return [
        f"rgmin {rgmin_version}, git {_git_sha()}",
        f"ASE {ase.__version__}, NumPy {np.__version__}, Python {platform.python_version()}",
        f"{platform.platform()}, {_cpu()}, {os.cpu_count()} logical CPUs",
    ]


def summarize(runs: list[dict]) -> dict:
    first = runs[0]
    for run in runs[1:]:
        same = (run["steps"], run["calls"], run["converged"]) == (
            first["steps"],
            first["calls"],
            first["converged"],
        )
        if not same or abs(run["energy"] - first["energy"]) > 1e-9:
            raise SystemExit(f"not repeatable: {first} vs {run}")
    walls = [run["wall"] for run in runs]
    return {**first, "median": statistics.median(walls), "min": min(walls)}


def write_table(env: list[str], rows: list[dict], repeats: int, steps: int) -> str:
    lines = [
        "# rgmin.optimize against ase.optimize",
        "",
        *[f"- {line}" for line in env],
        f"- fmax {FMAX}, step budget {steps}, {repeats} serial repeats per row,",
        "  default constructor arguments on both sides.",
        "",
        "Steps and force calls are identical on every repeat. Wall time is",
        "the whole `run`, Python calculator included, in milliseconds.",
        "ODE12r has no rgmin counterpart; its rows are ASE only.",
        "",
        "| case | optimizer | library | steps | force calls | energy | fmax | median ms | min ms | converged |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        flag = "yes" if row["converged"] else ("error" if row["error"] else "no")
        lines.append(
            f"| {row['case']} | {row['optimizer']} | {row['library']} | {row['steps']} | "
            f"{row['calls']} | {row['energy']:.6f} | {row['fmax']:.3e} | "
            f"{row['median'] * 1e3:.1f} | {row['min'] * 1e3:.1f} | {flag} |"
        )
    errors = [row for row in rows if row["error"]]
    if errors:
        lines.extend(["", "Errors:", ""])
        lines.extend(f"- {row['case']} {row['library']} {row['optimizer']}: {row['error']}" for row in errors)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--case", action="append", choices=list(CASES))
    args = parser.parse_args()

    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = environment()
    log = (raw / "runs.tsv").open("w")
    log.write("# " + "\n# ".join(env) + "\n")
    columns = ("case", "library", "optimizer", "repeat", "steps", "calls", "energy", "fmax", "converged", "wall", "error")
    log.write("\t".join(columns) + "\n")
    print("\n".join(env), flush=True)

    arms = [(name, library) for name in PAIRS for library in ("ase", "rgmin")]
    arms += [(name, "ase") for name in REFERENCE]
    rows = []
    for case in args.case or list(CASES):
        for name, library in arms:
            runs = []
            for repeat in range(args.repeats):
                run = run_once(case, library, name, args.steps)
                runs.append(run)
                log.write(
                    "\t".join(
                        str(value)
                        for value in (
                            case,
                            library,
                            name,
                            repeat,
                            run["steps"],
                            run["calls"],
                            repr(run["energy"]),
                            repr(run["fmax"]),
                            run["converged"],
                            repr(run["wall"]),
                            run["error"],
                        )
                    )
                    + "\n"
                )
                log.flush()
            row = summarize(runs)
            rows.append(row)
            print(
                f"{case:15} {name:16} {library:6} {row['steps']:6d} {row['calls']:6d} "
                f"{row['energy']:14.6f} {row['fmax']:10.3e} {row['median'] * 1e3:10.1f} "
                f"{row['min'] * 1e3:10.1f} {row['converged']} {row['error']}",
                flush=True,
            )
    log.close()
    table = write_table(env, rows, args.repeats, args.steps)
    (OUT / "results.md").write_text(table)


if __name__ == "__main__":
    main()
