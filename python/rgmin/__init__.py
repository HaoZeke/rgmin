# MIT License
# Copyright 2026 Rohit Goswami
"""ASE-shaped optimizers over the rgmin session.

Importing this module does not import ASE. The optimizer classes do,
on construction, because they are :class:`ase.optimize.optimize.Optimizer`
subclasses: ``trajectory``, ``logfile``, ``attach`` and ``run(fmax, steps)``
are ASE's.
"""

from __future__ import annotations

from rgmin._core import LbfgsPairs, LineSearch, Session, version

__all__ = ["BFGS", "FIRE", "LBFGS", "LbfgsPairs", "LineSearch", "Session", "__version__"]
__version__ = version


def __getattr__(name: str):
    if name in {"BFGS", "FIRE", "LBFGS"}:
        from rgmin import optimize

        return getattr(optimize, name)
    raise AttributeError(f"module 'rgmin' has no attribute {name}")
