"""Local minimization timings and memray peaks.

This commit has a C minimize entry and no Python extension module.
Each recorded sample calls that entry through a small shared library.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

_LIB = None
_LBFGS = 3
_FIRE = 17


def _library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "build" / "librgmin_asv.so", Path.cwd() / "build" / "librgmin_asv.so"]
    for base in (root / "env", root / ".asv" / "env"):
        if base.is_dir():
            candidates.extend(base.glob("*/project/build/librgmin_asv.so"))
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError("librgmin_asv.so was not found in build/")
    lib = ctypes.CDLL(str(path))
    lib.rgmin_asv_quadratic.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.rgmin_asv_quadratic.restype = ctypes.c_int
    _LIB = lib
    return lib


def _code(method: str) -> int:
    if method == "lbfgs":
        return _LBFGS
    if method == "fire":
        return _FIRE
    raise ValueError(method)


class QuadraticMinimize:
    """Wall time of one C minimize on a convex quadratic.

    ``number`` repeats that call inside one sample. asv divides the
    sample back to one call. Each call starts from the same numeric line.
    """

    params = ([16, 64], ["lbfgs", "fire"])
    param_names = ["n", "method"]
    rounds = 10
    repeat = 1
    number = 2000
    sample_time = 0.05
    warmup_time = 1.0
    min_run_count = 1
    timeout = 600

    def setup(self, n: int, method: str) -> None:
        _library()

    def time_minimize(self, n: int, method: str) -> None:
        if _library().rgmin_asv_quadratic(int(n), _code(method)) != 0:
            raise RuntimeError("minimize failed")


class QuadraticMemory:
    """Peak bytes from asv_bench_memray (methods named ray_*)."""

    params = ([16, 64], ["lbfgs", "fire"])
    param_names = ["n", "method"]
    rounds = 2
    number = 1
    repeat = 1
    timeout = 180

    def setup(self, n: int, method: str) -> None:
        _library()

    def ray_minimize(self, n: int, method: str) -> None:
        if _library().rgmin_asv_quadratic(int(n), _code(method)) != 0:
            raise RuntimeError("minimize failed")
