#!/usr/bin/env python3
"""Run the optimizer coverage benchmark without warm-up or retries."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    system: str
    method: str
    implementation: str
    seed: int
    round_index: int


class TimedPotential:
    def __init__(self, potential: Any) -> None:
        self.potential = potential
        self.calls = 0
        self.nanoseconds = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.potential, name)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return self.potential.evaluate(*args, **kwargs)
        finally:
            self.nanoseconds += time.perf_counter_ns() - started
            self.calls += 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return self.potential(*args, **kwargs)
        finally:
            self.nanoseconds += time.perf_counter_ns() - started
            self.calls += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--thread-count", type=int, required=True)
    parser.add_argument("--fmax", type=float, required=True)
    parser.add_argument("--stress-max", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--energy-tolerance", type=float, required=True)
    parser.add_argument("--rgmin-sha", required=True)
    parser.add_argument("--rgpot-sha", required=True)
    parser.add_argument("--rgpot-patch-sha", required=True)
    parser.add_argument("--eon-sha", required=True)
    parser.add_argument("--eon-rgpot-sha", required=True)
    parser.add_argument("--rgmin-wheel-sha256", required=True)
    parser.add_argument("--rgpot-wheel-sha256", required=True)
    parser.add_argument("--eon-wheel-sha256", required=True)
    parser.add_argument("--component", action="append", required=True)
    parser.add_argument("--rust-version", required=True)
    parser.add_argument("--cxx-version", required=True)
    return parser.parse_args()


def set_affinity(spec: str, thread_count: int) -> list[int]:
    cores = sorted({int(item) for item in spec.split(",")})
    if not cores or thread_count != len(cores):
        raise ValueError("thread-count must equal the number of pinned cores")
    available = os.sched_getaffinity(0)
    if any(core not in available for core in cores):
        raise ValueError("a requested core is outside the process affinity")
    os.sched_setaffinity(0, cores)
    return cores


def installed_packages() -> dict[str, str]:
    rows: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            rows[name] = dist.version
    return dict(sorted(rows.items(), key=lambda item: item[0].lower()))


def components(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        name, separator, version = item.partition("=")
        if not separator or not name or not version:
            raise ValueError("components must use name=version")
        parsed[name] = version
    return dict(sorted(parsed.items()))


def load_cluster(path: Path, seed: int) -> Atoms:
    import numpy as np
    from ase import Atoms

    positions = np.loadtxt(path, dtype=np.float64)
    rng = np.random.default_rng(seed)
    displacement = rng.normal(0.0, 0.02, positions.shape)
    displacement -= displacement.mean(axis=0)
    positions = positions + displacement
    return Atoms(
        numbers=np.full(len(positions), 18),
        positions=positions,
        cell=np.eye(3) * 100.0,
        pbc=False,
    )


def make_system(name: str, seed: int, fixtures: Path) -> tuple[Atoms, str, bool]:
    import numpy as np
    from ase.build import bulk

    if name == "LJ38":
        return load_cluster(fixtures / "lj38.xyz", seed), "lj", False
    if name == "LJ75":
        return load_cluster(fixtures / "lj75.xyz", seed), "lj", False
    if name == "Cu255-vacancy":
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((4, 4, 4))
        centre = np.asarray(atoms.cell).sum(axis=0) * 0.5
        vacancy = int(np.linalg.norm(atoms.positions - centre, axis=1).argmin())
        del atoms[vacancy]
        rng = np.random.default_rng(12001)
        atoms.positions += rng.normal(0.0, 0.02, atoms.positions.shape)
        return atoms, "morse-cu", False
    if name == "Si64":
        atoms = bulk("Si", "diamond", a=5.431, cubic=True).repeat((2, 2, 2))
        rng = np.random.default_rng(12002)
        atoms.positions += rng.normal(0.0, 0.02, atoms.positions.shape)
        return atoms, "sw", False
    if name == "Cu108-cell":
        atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3))
        atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)
        return atoms, "morse-cu", True
    if name == "Si64-cell":
        atoms = bulk("Si", "diamond", a=5.431, cubic=True).repeat((2, 2, 2))
        atoms.set_cell(atoms.cell * 1.02, scale_atoms=True)
        return atoms, "sw", True
    raise ValueError(f"unknown system {name}")


def tracked_calculator(kind: str, cell_relaxation: bool) -> tuple[RgpotCalculator, TimedPotential]:
    import rgpot
    from rgpot.ase import RgpotCalculator

    constructors = {
        "lj": getattr(rgpot, "LJPot", None),
        "morse": getattr(rgpot, "MorsePot", None),
        "sw": getattr(rgpot, "StillingerWeberSi", None),
        "emt": getattr(rgpot, "EMTPot", None),
    }
    if kind == "morse-cu":
        potential = rgpot.MorsePot(
            De=0.3429,
            a=1.3588,
            re=2.866,
            cutoff=5.0,
            switch_width=1.0,
        )
    else:
        constructor = constructors[kind]
        potential = None if constructor is None else constructor()
    if potential is None:
        raise RuntimeError(f"{kind} is unavailable in this rgpot build")
    tracked = TimedPotential(potential)
    return RgpotCalculator(tracked, numerical_stress=cell_relaxation), tracked


def fixed_optimizer(implementation: str, method: str, atoms: Atoms) -> Any:
    from ase.optimize import FIRE as AseFIRE
    from ase.optimize import LBFGS as AseLBFGS
    from rgmin.ase import FIRE2 as RgminFIRE2
    from rgmin.ase import LBFGS as RgminLBFGS
    from rgmin.ase import SD as RgminSD

    if implementation == "rgmin" and method == "lbfgs":
        return RgminLBFGS(atoms, logfile=None)
    if implementation == "rgmin" and method == "fire":
        return RgminFIRE2(atoms, logfile=None)
    if implementation == "rgmin" and method == "sd":
        return RgminSD(atoms, logfile=None)
    if implementation == "ase" and method == "lbfgs":
        return AseLBFGS(atoms, logfile=None)
    if implementation == "ase" and method == "fire":
        return AseFIRE(atoms, logfile=None)
    raise ValueError(f"unsupported fixed optimizer {implementation}/{method}")


def final_observables(atoms: Atoms, cell_relaxation: bool) -> tuple[float, float, float | None]:
    import numpy as np

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    fmax = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
    stress = None
    if cell_relaxation:
        stress = float(np.abs(atoms.get_stress(voigt=True)).max(initial=0.0))
    return energy, fmax, stress


def run_ase_family(
    task: Task,
    atoms: Atoms,
    cell_relaxation: bool,
    fmax: float,
    stress_max: float,
    max_steps: int,
) -> dict[str, Any]:
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS as AseLBFGS
    from rgmin.ase import LBFGS as RgminLBFGS

    if cell_relaxation:
        filtered = FrechetCellFilter(atoms)
        optimizer = (
            RgminLBFGS(filtered, logfile=None)
            if task.implementation == "rgmin"
            else AseLBFGS(filtered, logfile=None)
        )
        converged = False
        energy = float("nan")
        measured_fmax = float("inf")
        measured_stress = float("inf")
        for _ in optimizer.irun(fmax=0.0, steps=max_steps):
            energy, measured_fmax, measured_stress = final_observables(atoms, True)
            if measured_fmax <= fmax and measured_stress is not None and measured_stress <= stress_max:
                converged = True
                break
    else:
        optimizer = fixed_optimizer(task.implementation, task.method, atoms)
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        energy, measured_fmax, measured_stress = final_observables(atoms, False)
    return {
        "converged": converged,
        "energy_ev": energy,
        "fmax_ev_per_angstrom": measured_fmax,
        "stress_max_ev_per_angstrom3": measured_stress,
        "steps": int(optimizer.nsteps),
    }


def run_eon(
    task: Task,
    atoms: Atoms,
    fmax: float,
    max_steps: int,
) -> dict[str, Any]:
    import numpy as np
    import pyeonclient as eon

    methods = {
        "lbfgs": eon.OptType.LBFGS,
        "fire": eon.OptType.FIRE,
        "sd": eon.OptType.SD,
    }
    parameters = eon.Parameters()
    parameters.opt_method = methods[task.method]
    parameters.opt_converged_force = fmax
    parameters.opt_convergence_metric = "max_atom"
    parameters.opt_max_iterations = max_steps
    matter = eon.ase_to_matter(atoms, parameters=parameters)
    matter.reset_force_calls()
    relaxed, converged = matter.relax(inplace=False, quiet=True)
    energy = float(relaxed.potential_energy)
    forces = np.asarray(relaxed.forces, dtype=np.float64)
    measured_fmax = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
    return {
        "converged": bool(converged) and measured_fmax <= fmax,
        "energy_ev": energy,
        "fmax_ev_per_angstrom": measured_fmax,
        "stress_max_ev_per_angstrom3": None,
        "steps": None,
        "eon_force_calls": int(relaxed.force_calls),
    }


def run_task(
    task: Task,
    fixtures: Path,
    fmax: float,
    stress_max: float,
    max_steps: int,
) -> dict[str, Any]:
    atoms, potential_kind, cell_relaxation = make_system(task.system, task.seed, fixtures)
    calculator, tracked = tracked_calculator(potential_kind, cell_relaxation)
    atoms.calc = calculator
    started = time.perf_counter_ns()
    try:
        if task.implementation == "eon":
            outcome = run_eon(task, atoms, fmax, max_steps)
        else:
            outcome = run_ase_family(
                task,
                atoms,
                cell_relaxation,
                fmax,
                stress_max,
                max_steps,
            )
        error = None
    except Exception as exc:
        outcome = {
            "converged": False,
            "energy_ev": None,
            "fmax_ev_per_angstrom": None,
            "stress_max_ev_per_angstrom3": None,
            "steps": None,
        }
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter_ns() - started
    potential_ns = tracked.nanoseconds
    return {
        "system": task.system,
        "method": task.method,
        "implementation": task.implementation,
        "seed": task.seed,
        "round": task.round_index,
        "order_direction": "forward" if task.round_index % 2 == 0 else "reverse",
        "cell_relaxation": cell_relaxation,
        "wall_seconds": elapsed / 1.0e9,
        "potential_seconds": potential_ns / 1.0e9,
        "outside_potential_seconds": max(0, elapsed - potential_ns) / 1.0e9,
        "force_calls": tracked.calls,
        "error": error,
        **outcome,
    }


def task_cells() -> list[tuple[str, str, list[str], list[int]]]:
    fixed = [
        ("LJ38", list(range(10))),
        ("LJ75", list(range(10))),
        ("Cu255-vacancy", [0]),
        ("Si64", [0]),
    ]
    cells: list[tuple[str, str, list[str], list[int]]] = []
    for system, seeds in fixed:
        cells.extend(
            [
                (system, "lbfgs", ["rgmin", "ase", "eon"], seeds),
                (system, "fire", ["rgmin", "ase", "eon"], seeds),
                (system, "sd", ["rgmin", "eon"], seeds),
            ]
        )
    for system in ("Cu108-cell", "Si64-cell"):
        cells.append((system, "cell-lbfgs", ["rgmin", "ase"], [0]))
    return cells


def schedule(rounds: int) -> list[Task]:
    tasks: list[Task] = []
    cells = task_cells()
    for round_index in range(rounds):
        ordered_cells = cells if round_index % 2 == 0 else list(reversed(cells))
        for system, method, implementations, seeds in ordered_cells:
            ordered_impl = implementations if round_index % 2 == 0 else list(reversed(implementations))
            ordered_seeds = seeds if round_index % 2 == 0 else list(reversed(seeds))
            for seed in ordered_seeds:
                for implementation in ordered_impl:
                    tasks.append(Task(system, method, implementation, seed, round_index))
    return tasks


def write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="logm result may be inaccurate",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="Log flushes by default now",
    )
    if args.rounds < 12:
        raise ValueError("rounds must be at least 12")
    if args.output.exists():
        raise FileExistsError("output already exists; benchmark runs cannot be resumed or replaced")
    cores = set_affinity(args.cores, args.thread_count)
    tasks = schedule(args.rounds)
    document: dict[str, Any] = {
        "schema": 1,
        "protocol": {
            "rounds": args.rounds,
            "cores": cores,
            "thread_count": args.thread_count,
            "fmax_ev_per_angstrom": args.fmax,
            "stress_max_ev_per_angstrom3": args.stress_max,
            "max_steps": args.max_steps,
            "energy_agreement_tolerance_ev": args.energy_tolerance,
            "prewarm": False,
            "rescue_reruns": False,
            "order": "forward on even rounds, reverse on odd rounds",
            "cluster_perturbation_standard_deviation_angstrom": 0.02,
            "potentials": {
                "LJ38": "LJPot defaults",
                "LJ75": "LJPot defaults",
                "Cu255-vacancy": (
                    "MorsePot De=0.3429, a=1.3588, re=2.866, "
                    "cutoff=5.0, switch_width=1.0"
                ),
                "Si64": "StillingerWeberSi defaults",
                "Cu108-cell": (
                    "MorsePot De=0.3429, a=1.3588, re=2.866, "
                    "cutoff=5.0, switch_width=1.0"
                ),
                "Si64-cell": "StillingerWeberSi defaults",
            },
        },
        "sources": {
            "rgmin": args.rgmin_sha,
            "rgpot": args.rgpot_sha,
            "rgpot_binding_patch": args.rgpot_patch_sha,
            "eon": args.eon_sha,
            "eon_bundled_rgpot": args.eon_rgpot_sha,
        },
        "artifacts": {
            "rgmin_wheel_sha256": args.rgmin_wheel_sha256,
            "rgpot_wheel_sha256": args.rgpot_wheel_sha256,
            "eon_wheel_sha256": args.eon_wheel_sha256,
        },
        "build": {
            "profile": "release",
            "rust": args.rust_version,
            "cxx": args.cxx_version,
            "native_arch": False,
            "fast_math": False,
            "lto": False,
            "vector_extensions": False,
            "components": components(args.component),
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "packages": installed_packages(),
        },
        "gaps": [
            {
                "implementation": "ase",
                "method": "sd",
                "reason": "No steepest-descent comparator was requested for ASE.",
            },
            {
                "implementation": "eon",
                "method": "cell-lbfgs",
                "reason": "The eOn client optimizer does not expose variable-cell relaxation.",
            },
        ],
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, document)
    for index, task in enumerate(tasks, start=1):
        record = run_task(task, args.fixtures, args.fmax, args.stress_max, args.max_steps)
        document["records"].append(record)
        write_json(args.output, document)
        print(
            f"{index}/{len(tasks)} {task.system} {task.method} {task.implementation} "
            f"seed={task.seed} round={task.round_index} converged={record['converged']} "
            f"calls={record['force_calls']} wall={record['wall_seconds']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
