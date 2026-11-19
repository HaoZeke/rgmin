# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""Select OpenBLAS ARMV8 when an aarch64 SME kernel miscomputes dgemm."""

from __future__ import annotations

import os
import platform
import sys

import pytest

_REEXEC_GUARD = "_RGMIN_OPENBLAS_ARMV8_REEXEC"


def _dgemm_four_row_sum_broken() -> bool:
    import numpy as np

    ones = np.ones((4, 3), dtype=np.float64)
    ramp = np.arange(1.0, 5.0, dtype=np.float64)[:, None] * ones
    return float((ones.T @ ramp)[0, 0]) != 10.0


def pytest_configure(config: pytest.Config) -> None:
    del config
    if sys.platform != "linux" or platform.machine() != "aarch64":
        return
    if os.environ.get("OPENBLAS_CORETYPE") or os.environ.get(_REEXEC_GUARD):
        return
    if not _dgemm_four_row_sum_broken():
        return
    env = os.environ.copy()
    env["OPENBLAS_CORETYPE"] = "ARMV8"
    env[_REEXEC_GUARD] = "1"
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
