"""ASE's optimizer tests, on the rgmin classes.

The shape follows ``ase/test/optimize/test_optimizers.py``: rattle a
cell, ``run(fmax)`` must return true, the max force must be under
``fmax``, and ``converged`` agrees with an explicit force array.
One step at a tiny ``fmax`` must not report convergence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from rgmin import BFGS, FIRE, LBFGS
from rgmin._core import Session


def _atoms():
    atoms = bulk("Au") * (2, 2, 2)
    atoms.calc = EMT()
    atoms.rattle(stdev=0.1, seed=7)
    return atoms


@pytest.mark.parametrize("optcls", [BFGS, LBFGS, FIRE])
def test_optimize(optcls, tmp_path: Path):
    atoms = _atoms()
    ref = bulk("Au")
    ref.calc = EMT()
    ref_energy = ref.get_potential_energy()
    seen: list[float] = []
    with optcls(atoms, trajectory=str(tmp_path / "opt.traj"), logfile=str(tmp_path / "opt.log")) as opt:
        opt.attach(lambda: seen.append(float(atoms.get_potential_energy())))
        assert opt.run(fmax=0.05)
    forces = atoms.get_forces()
    fmax = float(np.max(np.linalg.norm(forces, axis=1)))
    assert fmax < 0.05
    assert opt.converged()
    assert opt.converged(forces=forces)
    per_atom = atoms.get_potential_energy() * len(ref) / len(atoms)
    assert abs(per_atom - ref_energy) < 1.0e-3
    assert seen
    assert (tmp_path / "opt.traj").is_file()
    assert (tmp_path / "opt.log").is_file()


@pytest.mark.parametrize("optcls", [BFGS, LBFGS, FIRE])
def test_one_step_is_not_converged(optcls):
    atoms = _atoms()
    with optcls(atoms, logfile=None) as opt:
        assert not opt.run(fmax=1e-9, steps=1)
    assert not opt.converged()


def test_lbfgs_signature_accepts_the_ase_keywords():
    atoms = _atoms()
    opt = LBFGS(
        atoms,
        logfile=None,
        maxstep=0.2,
        memory=8,
        damping=1.0,
        alpha=70.0,
        use_line_search=False,
    )
    assert opt.maxstep == 0.2
    assert opt.memory == 8
    opt.run(fmax=0.05, steps=5)


def test_eon_cap_escapes_the_well_the_ase_scale_keeps():
    """The scaled step stops near x0 = -1. The eOn cap does not."""

    def rosenbrock(x: list[float]) -> tuple[float, list[float]]:
        f = 0.0
        g = [0.0] * len(x)
        for i in range(len(x) - 1):
            a = 1.0 - x[i]
            b = x[i + 1] - x[i] * x[i]
            f += a * a + 100.0 * b * b
            g[i] += -2.0 * a - 400.0 * x[i] * b
            g[i + 1] += 200.0 * b
        return f, g

    start = [(-1.2 if i % 2 == 0 else 1.0) for i in range(10)]

    def relax(eon_cap: bool) -> float:
        session = Session("lbfgs", 10, max_atom=0.2, memory=8, eon_cap=eon_cap)
        x = list(start)
        value = rosenbrock(x)[0]
        for _ in range(20_000):
            x, value, gnorm = session.step(x, rosenbrock)
            if gnorm < 1e-5:
                break
        return value

    scaled = relax(False)
    capped = relax(True)
    assert scaled > 1.0
    assert capped < 1e-8
