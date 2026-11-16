# MIT License
# Copyright 2026 Rohit Goswami
"""Names that only the maturin build of ``rgmin._core`` defines.

``_core.pyi`` is the stub nanobind generates for the wheel's module, so
these names are typed in ``_maturin.pyi``. Importing this module from
the nanobind wheel raises ``ImportError``.
"""

from rgmin._core import LbfgsPairs  # pyright: ignore[reportAttributeAccessIssue]
from rgmin._core import LineSearch  # pyright: ignore[reportAttributeAccessIssue]
from rgmin._core import RiemannianManifold  # pyright: ignore[reportAttributeAccessIssue]
from rgmin._core import Session  # pyright: ignore[reportAttributeAccessIssue]
from rgmin._core import lowest_eigenpairs  # pyright: ignore[reportAttributeAccessIssue]
from rgmin._core import version  # pyright: ignore[reportAttributeAccessIssue]

__all__ = [
    "LbfgsPairs",
    "LineSearch",
    "RiemannianManifold",
    "Session",
    "lowest_eigenpairs",
    "version",
]
