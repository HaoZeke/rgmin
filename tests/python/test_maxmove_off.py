# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""A step cap that is not a positive finite length is no cap.

``maxmove`` 0 and -0 keep the defaults. A negative ``maxmove`` turns
every cap off, the default per-atom cap included. A non-positive
``maxstep`` clears the per-atom cap, and on the default L-BFGS step
restores 0.2. A non-finite value is a ``ValueError``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import rgmin

WEIGHT = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
START = np.array([3.0, -4.0, 1.5, 2.5, -2.0, 0.5])
METHODS = [
    "lbfgs",
    "bfgs",
    "fire",
    "fire2",
    "quickmin",
    "ode12r",
    "sd",
    "cg",
    "polak-ribiere-plus",
    "bb",
    "sr1",
    "sr2",
    "adam",
    "diis",
    "newton-cg",
]
HESSIAN_METHODS = ["dogleg", "newton", "rfo"]


def _bowl(x: np.ndarray) -> tuple[float, np.ndarray]:
    return 0.5 * float(WEIGHT @ (x * x)), WEIGHT * x


def _run(method: str, **options: float) -> tuple[float, int, list[float]]:
    x = START.copy()
    report, out = rgmin.minimize(_bowl, x, method=method, gtol=1e-14, maxiter=6, **options)
    return report.value, report.steps, out.tolist()


@pytest.mark.parametrize("method", METHODS)
def test_maxmove_classes(method: str) -> None:
    keep = _run(method, maxmove=0.0)
    assert _run(method, maxmove=-0.0) == keep
    off = _run(method, maxmove=-1.0)
    for cap in (-1e-3, -1e9):
        assert _run(method, maxmove=cap) == off, cap
    assert off[0] <= _bowl(START)[0]


@pytest.mark.parametrize("method", METHODS)
def test_maxstep_classes(method: str) -> None:
    cleared = _run(method, maxstep=0.0)
    for cap in (-0.0, -1.0, -1e9):
        assert _run(method, maxstep=cap) == cleared, cap
        assert _run(method, atom_maxmove=cap) == cleared, cap


def _run_hess(method: str, **options: float) -> tuple[list[float], list[float]]:
    solver = rgmin.Solver(START.size, rgmin.SolverConfig(method, **options))
    x = START.copy()
    values = [solver.step_hess(_bowl, lambda _: np.diag(WEIGHT), x).value for _ in range(6)]
    return values, x.tolist()


@pytest.mark.parametrize("method", HESSIAN_METHODS)
def test_hessian_maxmove_classes(method: str) -> None:
    keep = _run_hess(method, maxmove=0.0)
    assert _run_hess(method, maxmove=-0.0) == keep
    off = _run_hess(method, maxmove=-1.0)
    for cap in (-1e-3, -1e9):
        assert _run_hess(method, maxmove=cap) == off, cap
    assert off[0][0] <= _bowl(START)[0]


@pytest.mark.parametrize("key", ["maxmove", "maxstep", "atom_maxmove"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_cap_is_refused(key: str, value: float) -> None:
    with pytest.raises(ValueError):
        rgmin.SolverConfig("lbfgs", **{key: value})


def test_a_negative_maxmove_steps_downhill() -> None:
    f0, g0 = _bowl(START)
    for method in METHODS:
        x = START.copy()
        report, _ = rgmin.minimize(_bowl, x, method=method, maxiter=1, maxmove=-1.0)
        assert report.value <= f0, method
        assert float((x - START) @ g0) <= 0.0, method


def test_ase_optimizers_take_a_non_positive_cap_as_no_cap() -> None:
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.emt import EMT

    from rgmin.ase import BFGS, FIRE, FIRE2, LBFGS, SD, QuickMin

    a = 2.7
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0, a],
            [0.0, 0.0, -a],
        ]
    )
    rattle = np.random.default_rng(7).normal(scale=0.05, size=base.shape)

    def relax(cls: type, **options: float) -> list[float]:
        atoms = Atoms("Pt7", positions=base + rattle)
        atoms.calc = EMT()
        opt = cls(atoms, logfile=None, **options)
        opt.run(fmax=1e-9, steps=6)
        return atoms.get_positions().ravel().tolist()

    for cls in (LBFGS, BFGS, FIRE, FIRE2, SD, QuickMin):
        cleared = relax(cls, maxstep=0.0)
        for cap in (-0.0, -1.0):
            assert relax(cls, maxstep=cap) == cleared, (cls.__name__, cap)
        off = relax(cls, maxmove=-1.0)
        assert relax(cls, maxmove=-1e9) == off, cls.__name__
