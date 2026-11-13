#!/usr/bin/env python3
"""Reference points from SciPy L-BFGS-B for ``Lbfgs::minimize_box``.

stdin JSON: {"problem": "separable"|"coupled"|"rosenbrock",
             "maxcor": int, "gtol": float, "ftol": float}

The three problems are the same literals as ``tests/lbfgs_box_scipy.rs``.
``coupled`` is not separable: the bound-constrained minimizer is not the
box projection of the free minimizer, and two bounds are active.

stdout JSON: {"x", "fun", "nfev", "success", "proj"}
``proj`` is the infinity norm of the projected gradient.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import minimize


def separable():
    scales = np.array([1.0, 4.0, 9.0, 16.0, 0.5, 2.0])
    targets = np.array([2.5, -1.0, 0.3, 8.0, 1.2, -3.0])
    lo = np.array([0.0, -2.0, -1.0, 0.0, -5.0, -1.0])
    hi = np.array([1.0, 2.0, 1.0, 3.0, 5.0, 0.5])
    x0 = np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0])

    def fun(x):
        d = x - targets
        return float(0.5 * np.dot(scales, d * d))

    def jac(x):
        return scales * (x - targets)

    return fun, jac, x0, lo, hi


def coupled():
    # Tridiagonal Hessian, 4 on the diagonal and 1 on the off-diagonals.
    # The free minimizer is ``t``; the box cuts four coordinates, and the
    # fifth moves off ``t`` because of the coupling.
    t = np.array([3.0, -2.0, 0.2, 4.0, -3.0])
    n = t.size
    h = np.eye(n) * 4.0
    idx = np.arange(n - 1)
    h[idx, idx + 1] = 1.0
    h[idx + 1, idx] = 1.0
    b = h @ t
    lo = np.array([0.0, -1.0, -2.0, 0.0, -1.0])
    hi = np.array([1.5, 1.0, 2.0, 1.0, 1.0])
    x0 = np.zeros(n)

    def fun(x):
        return float(0.5 * x @ h @ x - b @ x)

    def jac(x):
        return h @ x - b

    return fun, jac, x0, lo, hi


def rosenbrock():
    # The free minimum (1, 1) lies past the upper bound on the first coordinate.
    lo = np.array([-2.0, -1.0])
    hi = np.array([0.5, 2.0])
    x0 = np.array([-1.2, 1.0])

    def fun(x):
        a, b = x
        return float((1.0 - a) ** 2 + 100.0 * (b - a * a) ** 2)

    def jac(x):
        a, b = x
        t = b - a * a
        return np.array([-2.0 * (1.0 - a) - 400.0 * a * t, 200.0 * t])

    return fun, jac, x0, lo, hi


PROBLEMS = {
    "separable": separable,
    "coupled": coupled,
    "rosenbrock": rosenbrock,
}


def projected_inf(x, g, lo, hi):
    pg = np.minimum(np.maximum(g, x - hi), x - lo)
    return float(np.max(np.abs(pg)))


def main() -> None:
    spec = json.load(sys.stdin)
    name = spec["problem"]
    if name not in PROBLEMS:
        raise SystemExit(f"unknown problem {name}")
    fun, jac, x0, lo, hi = PROBLEMS[name]()
    res = minimize(
        fun,
        x0,
        jac=jac,
        method="L-BFGS-B",
        bounds=list(zip(lo.tolist(), hi.tolist(), strict=True)),
        options={
            "maxcor": int(spec["maxcor"]),
            "gtol": float(spec["gtol"]),
            "ftol": float(spec["ftol"]),
            "maxiter": 500,
            "maxfun": 2000,
            "disp": False,
        },
    )
    x = np.asarray(res.x, dtype=float)
    out = {
        "x": [float(v) for v in x],
        "fun": float(res.fun),
        "nfev": int(res.nfev),
        "success": bool(res.success),
        "proj": projected_inf(x, jac(x), lo, hi),
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
