# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""A zero iteration budget evaluates the start and takes no step."""

from __future__ import annotations

import numpy as np
import pytest

import rgmin

_CURVATURE = np.array([1.0, 2.0, 3.0])


def _bowl(x: np.ndarray) -> tuple[float, np.ndarray]:
    g = _CURVATURE * x
    return 7.0 + 0.5 * float(x @ g), g


@pytest.mark.parametrize("method", ["lbfgs", "bfgs", "fire", "bb", "quickmin", "diis"])
@pytest.mark.parametrize("start", [[1.0, -1.0, 0.5], [0.0, 0.0, 0.0]])
def test_zero_budget_reports_the_evaluated_start(method: str, start: list[float]) -> None:
    x0 = np.array(start)
    value, grad = _bowl(x0)
    report, x = rgmin.minimize(_bowl, x0.copy(), method, maxiter=0, gtol=1e-8)
    np.testing.assert_array_equal(x, x0)
    assert report.steps == 0
    assert report.value == value
    assert report.grad_norm == pytest.approx(np.linalg.norm(grad), abs=1e-12)
    expected = rgmin.Stop.Tolerance if np.linalg.norm(grad) < 1e-8 else rgmin.Stop.Budget
    assert report.stop == expected


def test_zero_budget_applies_the_noise_floor() -> None:
    x0 = np.array([1e-6, 0.0, 0.0])
    report, _ = rgmin.minimize(_bowl, x0.copy(), "lbfgs", maxiter=0, gtol=1e-12, noise_eps=1e-6)
    assert report.grad_norm == pytest.approx(1e-6)
    assert report.stop == rgmin.Stop.NoiseFloor


def test_zero_budget_refuses_a_non_finite_gradient() -> None:
    def broken(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 1.0, np.array([np.nan, 0.0, 0.0])

    with pytest.raises(Exception, match="non-finite gradient"):
        rgmin.minimize(broken, np.zeros(3), "lbfgs", maxiter=0)
