# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Default per-atom cap, and an explicit maxstep that replaces it.

The start is the seed-1 packing from bench/atomistic lj38, scaled by
one half. The Rust test of the same geometry is tests/default_maxstep.rs.
maxstep keeps ASE's meaning: no atom moves farther than maxstep in one
step, and a longer step is scaled down whole.
"""

from __future__ import annotations

import numpy as np

import rgmin

N = 38
CAP = 0.2
BOUND_PAIR = 6.0
# The TSan job runs each test at 4 threads x 5 iterations inside a 120 s
# faulthandler budget, so the pair sums are NumPy rather than Python loops.
_I, _J = np.triu_indices(N, k=1)


def _compressed_lj38() -> np.ndarray:
    radius = 2.4
    state = (1 * 7919 + 17) & 0xFFFFFFFFFFFFFFFF

    def next_u64() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def uniform() -> float:
        return (next_u64() >> 11) / float(1 << 53)

    pos: list[float] = []
    while len(pos) < 3 * N:
        point = [(2.0 * uniform() - 1.0) * radius for _ in range(3)]
        if sum(value * value for value in point) > radius * radius:
            continue
        close = False
        for base in range(0, len(pos), 3):
            dist = sum((pos[base + k] - point[k]) ** 2 for k in range(3))
            if dist < 0.81:
                close = True
                break
        if not close:
            pos.extend(point)
    return np.asarray(pos, dtype=np.float64) * 0.5


def _lj(x: np.ndarray) -> tuple[float, np.ndarray]:
    d = x.reshape(N, 3)[_I] - x.reshape(N, 3)[_J]
    inv2 = 1.0 / np.sum(d * d, axis=1)
    sr6 = inv2 * inv2 * inv2
    sr12 = sr6 * sr6
    energy = float(np.sum(4.0 * (sr12 - sr6)))
    pair = (24.0 * inv2 * (sr6 - 2.0 * sr12))[:, None] * d
    grad = np.zeros((N, 3), dtype=np.float64)
    np.add.at(grad, _I, pair)
    np.subtract.at(grad, _J, pair)
    return energy, grad.reshape(-1)


def _max_pair(x: np.ndarray) -> float:
    d = x.reshape(N, 3)[_I] - x.reshape(N, 3)[_J]
    return float(np.sqrt(np.max(np.sum(d * d, axis=1))))


def _max_atom_step(before: np.ndarray, after: np.ndarray) -> float:
    d = (after - before).reshape(-1, 3)
    return float(np.max(np.sqrt(np.sum(d * d, axis=1))))


def _first_step(method: str, **options: float) -> np.ndarray:
    x = _compressed_lj38()
    before = x.copy()
    rgmin.Solver(x.size, rgmin.SolverConfig(method, maxiter=2, gtol=1e-12, **options)).step(_lj, x)
    return x - before


def _first_move(method: str, **options: float) -> float:
    step = _first_step(method, **options)
    return _max_atom_step(np.zeros_like(step), step)


def _soft_bowl(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Curvature 0.01, so the quasi-Newton step from far out is long."""
    return 0.005 * float(x @ x), 0.01 * x


def _moves(fg, x: np.ndarray, steps: int, **options: float) -> list[float]:
    solver = rgmin.Solver(x.size, rgmin.SolverConfig("lbfgs", maxiter=steps, gtol=1e-8, **options))
    moves = []
    for _ in range(steps):
        before = x.copy()
        report = solver.step(fg, x)
        moves.append(_max_atom_step(before, x))
        if report.stop != rgmin.Stop.Budget:
            break
    return moves


def test_default_cap_keeps_lj38_bound_and_maxstep_overrides() -> None:
    start = _compressed_lj38()
    assert _lj(start)[0] > 1.0e5
    assert _max_pair(start) < BOUND_PAIR
    x = start.copy()
    solver = rgmin.Solver(x.size, rgmin.SolverConfig("lbfgs", maxiter=800, gtol=1e-2))
    report = None
    peak = 0.0
    for _ in range(800):
        before = x.copy()
        report = solver.step(_lj, x)
        peak = max(peak, _max_atom_step(before, x))
        if report.grad_norm < 1e-2:
            break
    assert report is not None
    assert report.stop == rgmin.Stop.Tolerance
    assert report.value < -160.0, report.value
    assert _max_pair(x) <= BOUND_PAIR
    assert peak <= CAP + 1e-9, peak
    bfgs = _first_move("bfgs")
    assert bfgs > 1.0, bfgs
    cleared = _first_move("bfgs", maxstep=0.0)
    assert cleared > 1.0, cleared
    wider = _first_move("bfgs", maxstep=0.5)
    assert CAP < wider <= 0.5 + 1e-9, wider
    named = _first_move("bfgs", atom_maxmove=0.5)
    assert CAP < named <= 0.5 + 1e-9, named
    held = _first_move("bfgs", maxstep=CAP)
    assert 0.05 < held <= CAP + 1e-9, held


def test_lbfgs_maxstep_scales_a_long_first_step_like_ase() -> None:
    """ASE's maxstep: the longest atom move is maxstep, the direction kept.

    The first eOn step is ``H0`` times the force, with ``H0`` from one
    finite-difference probe. Uncapped it moves an atom 0.034 here, so
    both limits below have to act on it.
    """
    free = _first_step("lbfgs", maxstep=1.0e3)
    longest = _max_atom_step(np.zeros_like(free), free)
    for maxstep in (0.01, 0.02):
        assert longest > 1.5 * maxstep, (maxstep, longest)
        held = _first_step("lbfgs", maxstep=maxstep)
        assert abs(_max_atom_step(np.zeros_like(held), held) - maxstep) <= 1e-12 * maxstep
        assert np.allclose(held, free * (maxstep / longest), rtol=0.0, atol=1e-14)
        assert np.array_equal(held, _first_step("lbfgs", atom_maxmove=maxstep))


def test_lbfgs_maxstep_bounds_every_step() -> None:
    for maxstep in (0.01, 0.02):
        moves = _moves(_lj, _compressed_lj38(), 300, maxstep=maxstep)
        assert max(moves) <= maxstep * (1.0 + 1e-12), (maxstep, max(moves))
        assert sum(move > 0.999 * maxstep for move in moves) >= 10, (maxstep, moves[:20])
    free = _moves(_soft_bowl, np.full(9, 10.0), 1, maxstep=1.0e3)
    assert free[0] > 10.0, free
    for maxstep in (0.5, 3.0):
        moves = _moves(_soft_bowl, np.full(9, 10.0), 40, maxstep=maxstep)
        assert abs(moves[0] - maxstep) <= 1e-12 * maxstep, (maxstep, moves[0])
        assert max(moves) <= maxstep * (1.0 + 1e-12), (maxstep, moves)


def test_lbfgs_nonpositive_maxstep_restores_the_eon_cap() -> None:
    """The eOn step always has a per-atom limit. Non-positive means 0.2."""
    for maxstep in (0.0, -1.0):
        moves = _moves(_soft_bowl, np.full(9, 10.0), 5, maxstep=maxstep)
        assert all(abs(move - CAP) <= 1e-12 for move in moves), (maxstep, moves)
