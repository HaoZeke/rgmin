# MIT License
# Copyright 2026 Rohit Goswami
"""ASE-shaped optimizers over the rgmin session.

``rgmin.optimize`` holds every optimizer class; ``BFGS``, ``LBFGS``,
``LBFGSLineSearch``, ``FIRE`` and ``FIRE2`` are re-exported here.
Importing this module does not import ASE. The optimizer classes do,
on construction, because they are :class:`ase.optimize.optimize.Optimizer`
subclasses: ``trajectory``, ``logfile``, ``attach`` and ``run(fmax, steps)``
are ASE's.
"""

from __future__ import annotations

import numpy as np

from rgmin._core import LbfgsPairs, LineSearch, Session, lowest_eigenpairs as _lowest_eigenpairs, version

__all__ = [
    "BFGS",
    "FIRE",
    "FIRE2",
    "LBFGS",
    "LBFGSLineSearch",
    "LbfgsPairs",
    "LineSearch",
    "Session",
    "__version__",
    "lowest_eigenpairs",
]
__version__ = version
_OPTIMIZERS = frozenset(
    {
        "Adam",
        "BFGS",
        "BarzilaiBorwein",
        "CG",
        "ConjugateDescent",
        "DaiYuan",
        "EonCG",
        "FIRE",
        "FIRE2",
        "FRPR",
        "FletcherReeves",
        "HagerZhang",
        "HestenesStiefel",
        "LBFGS",
        "LBFGSLineSearch",
        "LiuStorey",
        "NewtonCG",
        "NewtonPCG",
        "PolakRibiere",
        "SR1",
        "SR2",
        "SteepestDescent",
    }
)


def lowest_eigenpairs(hvp, x0_block, *, x=None, tol=1e-8, max_iter=None):
    """Lowest eigenpairs of a symmetric product.

    ``hvp(x, v)`` returns ``H(x) v`` without forming ``H``.
    ``x0_block`` has shape ``(n, k)``, one column per starting vector.
    Block LOBPCG (Knyazev 2001) returns the values ascending and the
    vectors as the columns of an ``(n, k)`` array. ``x`` is the point
    where the product is taken; omitted, it is the origin.

    The default tolerance is ``1e-8``. ``max_iter`` defaults to
    ``max(n, 8)`` block expansions. A pair that misses the tolerance
    raises ``RuntimeError``. ``k = 0``, ``k`` above ``n``, and a
    non-finite value or product are ``ValueError``.
    """
    block = np.asarray(x0_block, dtype=np.float64)
    if block.ndim != 2:
        raise ValueError("x0_block must have shape (n, k)")
    point = None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)

    def product(point, vector):
        out = hvp(np.asarray(point, dtype=np.float64), np.asarray(vector, dtype=np.float64))
        return np.asarray(out, dtype=np.float64).reshape(-1).tolist()

    values, columns = _lowest_eigenpairs(
        product,
        block.tolist(),
        None if point is None else point.tolist(),
        tol=tol,
        max_iter=max_iter,
    )
    return np.asarray(values, dtype=np.float64), np.asarray(columns, dtype=np.float64).T


def __getattr__(name: str):
    if name in _OPTIMIZERS:
        from rgmin import optimize

        return getattr(optimize, name)
    raise AttributeError(f"module 'rgmin' has no attribute {name}")
