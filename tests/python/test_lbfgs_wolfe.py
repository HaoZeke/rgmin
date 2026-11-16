# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Strong Wolfe is an explicit L-BFGS opt-in: ``linesearch="wolfe"``.

The default L-BFGS step is the line-search-free eOn cap.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import rgmin

_SCALES = np.array([1.0, 4.0, 9.0, 16.0, 2.0, 3.0])
_X0 = (0.4, -0.2, 0.7, 0.1, -0.3, 0.5)
C1 = 1.0e-4
C2 = 0.9


def _quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
    return 0.5 * float(x @ (_SCALES * x)), _SCALES * x


def _run(steps: int, **options: object) -> tuple[list[np.ndarray], list[int]]:
    x = np.array(_X0, dtype=np.float64)
    calls = [0]

    def counted(z: np.ndarray) -> tuple[float, np.ndarray]:
        calls[0] += 1
        return _quadratic(z)

    solver = rgmin.Solver(x.size, rgmin.SolverConfig("lbfgs", maxiter=steps, gtol=1e-10, **options))
    points = [x.copy()]
    per_step = []
    for _ in range(steps):
        before = calls[0]
        solver.step(counted, x)
        per_step.append(calls[0] - before)
        points.append(x.copy())
    return points, per_step


def test_wolfe_steps_meet_the_strong_wolfe_conditions() -> None:
    points, _ = _run(8, linesearch="wolfe", maxstep=1.0e3)
    checked = 0
    for x0, x1 in zip(points, points[1:]):
        f0, g0 = _quadratic(x0)
        if f0 <= 1.0e-12 or np.array_equal(x0, x1):
            break
        f1, g1 = _quadratic(x1)
        step = x1 - x0
        slope = float(g0 @ step)
        assert slope < 0.0
        assert f1 <= f0 + C1 * slope
        assert abs(float(g1 @ step)) <= C2 * abs(slope)
        checked += 1
    assert checked >= 5, checked


def test_wolfe_is_not_the_default_step() -> None:
    eon, eon_calls = _run(4, maxstep=1.0e3)
    wolfe, wolfe_calls = _run(4, linesearch="wolfe", maxstep=1.0e3)
    assert not np.array_equal(eon[1], wolfe[1])
    # The eOn first step evaluates the start, its probe, and the trial.
    assert eon_calls[0] == 3, eon_calls
    assert wolfe_calls[0] == 2, wolfe_calls
    with pytest.raises(ValueError, match="has no effect on lbfgs"):
        _run(4, linesearch="brent", maxstep=1.0e3)


def test_maxstep_bounds_the_wolfe_steps() -> None:
    from test_maxstep import _compressed_lj38, _lj, _max_atom_step

    x = _compressed_lj38()
    solver = rgmin.Solver(x.size, rgmin.SolverConfig("lbfgs", maxiter=200, gtol=1e-2, linesearch="wolfe", maxstep=0.02))
    moves = []
    for _ in range(200):
        before = x.copy()
        report = solver.step(_lj, x)
        moves.append(_max_atom_step(before, x))
        if report.grad_norm < 1e-2:
            break
    assert max(moves) <= 0.02 * (1.0 + 1e-12), max(moves)
    assert sum(move > 0.999 * 0.02 for move in moves) >= 10


def test_ase_lbfgs_wolfe_matches_ase_calls() -> None:
    pytest.importorskip("ase")
    from ase.optimize import LBFGS as AseLBFGS
    from test_ase import _iteration_calls

    from rgmin.ase import LBFGS

    ours, theirs, moved = _iteration_calls(functools.partial(LBFGS, linesearch="wolfe"), AseLBFGS)
    assert moved
    assert ours == theirs
