#!/usr/bin/env python3
"""Minimal reproduction for capped L-BFGS on a variable cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.filters import FrechetCellFilter
from rgmin.ase import LBFGS

from run import final_observables, make_system, tracked_calculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("Cu108-cell", "Si64-cell"), required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--fmax", type=float, required=True)
    parser.add_argument("--stress-max", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atoms, kind, cell_relaxation = make_system(args.system, 0, args.fixtures)
    calculator, tracked = tracked_calculator(kind, cell_relaxation)
    atoms.calc = calculator
    initial = final_observables(atoms, True)
    optimizer = LBFGS(FrechetCellFilter(atoms), logfile=None)
    converged = False
    final = initial
    for _ in optimizer.irun(fmax=0.0, steps=args.steps):
        final = final_observables(atoms, True)
        if final[1] <= args.fmax and final[2] is not None and final[2] <= args.stress_max:
            converged = True
            break
    print(
        json.dumps(
            {
                "system": args.system,
                "converged": converged,
                "steps": optimizer.nsteps,
                "force_calls": tracked.calls,
                "initial_energy_ev": initial[0],
                "final_energy_ev": final[0],
                "initial_stress_max_ev_per_angstrom3": initial[2],
                "final_stress_max_ev_per_angstrom3": final[2],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
