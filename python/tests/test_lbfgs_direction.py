"""The L-BFGS direction takes a caller-supplied H0 and does not call an objective."""

import numpy as np
import pytest

from rgmin import LbfgsPairs


def dense(diag, pairs, g):
    h = np.diag(np.asarray(diag, dtype=float))
    for s, y in pairs:
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        rho = 1.0 / float(np.dot(s, y))
        eye = np.eye(s.size) - rho * np.outer(y, s)
        h = eye.T @ h @ eye + rho * np.outer(s, s)
    return -h @ np.asarray(g, dtype=float)


def test_empty_memory_is_minus_h0_g():
    pairs = LbfgsPairs(3)
    g = [1.0, -2.0, 0.5]
    diag = np.array([0.5, 0.25, 2.0])
    got = pairs.direction(g, diag)
    assert got == pytest.approx((-diag * g).tolist())
    assert pairs.direction(g, 0.5) == pytest.approx([-0.5, 1.0, -0.25])
    assert pairs.direction(g, lambda v: diag * np.asarray(v)) == pytest.approx(got)


def test_diagonal_matches_numpy_and_the_callback():
    s1 = [1.0, 0.5, -0.25]
    y1 = [2.0, 1.0, 0.5]
    s2 = [0.5, -0.25, 1.0]
    y2 = [1.0, 0.5, 2.0]
    g = [1.0, -2.0, 0.5]
    diag = np.array([0.5, 0.25, 2.0])
    pairs = LbfgsPairs(3, capacity=4)
    assert pairs.push(s1, y1)
    assert pairs.push(s2, y2)
    got = np.asarray(pairs.direction(g, diag))
    want = dense(diag, [(s1, y1), (s2, y2)], g)
    assert got == pytest.approx(want, rel=1e-10, abs=1e-10)
    via_callback = np.asarray(pairs.direction(g, lambda v: diag * np.asarray(v)))
    assert via_callback == pytest.approx(got)


def test_bad_lengths_raise():
    with pytest.raises(ValueError):
        LbfgsPairs(0)
    pairs = LbfgsPairs(2)
    with pytest.raises(ValueError):
        pairs.push([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        pairs.direction([1.0], 0.5)
    with pytest.raises(ValueError):
        pairs.direction([1.0, -1.0], np.array([1.0]))
    with pytest.raises(ValueError):
        pairs.direction([1.0, -1.0], lambda v: [v[0]])
    assert len(pairs) == 0
    assert pairs.push([1.0, 0.0], [-1.0, 0.0]) is False
