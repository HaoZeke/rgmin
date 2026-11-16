# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""SHA-256 of trajectories at several thread counts.

The LJ38 and Pt7 goldens pin ASE's LennardJones and EMT, which are
NumPy-dispatched. They are not pins of rgmin. rgmin's own results must
be bit-identical across ISAs with no environment variable. That check
is the diagonal quadratic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

# One hash per fixture. 3.14 and 3.14t both compare against these values.
# ASE LennardJones. Compared only in a process started with the pin below:
# its ``(sigma**2 / r2) ** 3`` goes through NumPy's AVX512F power kernel,
# which rounds differently, and the eOn step's finite-difference probe
# carries that last bit into every frame. Every platform agrees under it.
LJ38 = "9ff5ac4e272b58e733c2336e8b2580a566cfd0eefd77c3c1797359fb024cf24a"
# ASE EMT. Compared only in a process that was started with the pin below.
# Linux, macOS, and arm agree. Windows NumPy does not, on this trajectory.
PT7_EMT = "299f8164ab0df6f3b59731098b3d83e12c13c0b4117426437bf9af9df0a17ce8"
PT7_EMT_WINDOWS = "498757428117987ca8d99e3a209e83459f5b90b6196dba32ca695a534de0f64e"
# rgmin Solver, pure-Python diagonal quadratic, no NumPy pin.
# The solver uses the default per-atom cap of 0.2.
RGMIN_QUADRATIC = "b2ef984a0c4b84ba6c176428a2a0b3370b34d1fbb65ebe1126a6732a0cb62d7a"

# NumPy 2.2 still dispatches AVX512F and AVX512_ICL by those names.
# NumPy 2.5 dispatches X86_V3, X86_V4, and AVX512_ICL. A name that is
# not in this build's dispatch list makes the import warn, so the pin
# is the intersection, discovered before the golden process starts.
_STEPS = 5
_SCALES = (1.0, 4.0, 9.0, 16.0, 2.0, 3.0)
_QUADRATIC_X0 = (0.4, -0.2, 0.7, 0.1, -0.3, 0.5)

_PT7_THREADS = """
from test_invariance import assert_numpy_dispatch_disabled, pt7_hashes
assert_numpy_dispatch_disabled()
for workers in (1, 2, 8):
    hashes = pt7_hashes(workers)
    if len(set(hashes)) != 1:
        raise SystemExit(repr(hashes))
    print(hashes[0])
"""

_PT7_ONCE = """
from test_invariance import assert_numpy_dispatch_disabled, trajectory_sha256
assert_numpy_dispatch_disabled()
print(trajectory_sha256("pt7"))
"""

_LJ38_THREADS = """
from test_invariance import assert_numpy_dispatch_disabled, lj38_hashes
assert_numpy_dispatch_disabled()
for workers in (1, 2, 8):
    print(" ".join(lj38_hashes(workers)))
"""

_LJ38_ONCE = """
from test_invariance import assert_numpy_dispatch_disabled, trajectory_sha256
assert_numpy_dispatch_disabled()
print(trajectory_sha256("lj38"))
"""

_DISPATCH_PROBE = """
import json
from numpy._core._multiarray_umath import __cpu_dispatch__
print(json.dumps(list(__cpu_dispatch__)))
"""


def _blocked(name: str) -> bool:
    return name in {"X86_V3", "X86_V4"} or name.startswith("AVX512")


def assert_numpy_dispatch_disabled() -> None:
    """Confirm this process disabled NumPy's AVX-512 and X86_V3/V4 paths.

    ``numpy.show_runtime`` prints the enabled dispatch targets and
    returns None. The same split comes from ``__cpu_features__``. CPUID
    bits that are not dispatch targets stay enabled. This pins ASE's
    LennardJones and EMT, which are NumPy-dispatched, and not rgmin.
    """
    import contextlib
    import io
    import re

    import numpy as np
    from numpy._core._multiarray_umath import __cpu_dispatch__, __cpu_features__

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        shown = np.show_runtime()
    assert shown is None
    match = re.search(r"'found': \[(.*?)\]", buffer.getvalue(), re.DOTALL)
    assert match is not None
    found = re.findall(r"'([^']+)'", match.group(1))
    for name in found:
        assert not _blocked(name), name
    for name in __cpu_dispatch__:
        if _blocked(name):
            assert not __cpu_features__[name], name


def _lj38() -> object:
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    positions = np.random.default_rng(38).normal(scale=0.8, size=(38, 3))
    atoms = Atoms(numbers=[1] * 38, positions=positions)
    atoms.calc = LennardJones()
    return atoms


def _pt7() -> object:
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.emt import EMT

    a = 2.7
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0, a],
            [0.0, 0.0, -a],
        ]
    )
    atoms = Atoms("Pt7", positions=base + np.random.default_rng(7).normal(scale=0.02, size=base.shape))
    atoms.calc = EMT()
    return atoms


def trajectory_sha256(kind: str) -> str:
    from rgmin.ase import LBFGS

    atoms = _lj38() if kind == "lj38" else _pt7()
    frames: list[bytes] = []

    def grab() -> None:
        frames.append(np.array(atoms.get_positions(), dtype=np.float64).tobytes())

    opt = LBFGS(atoms, logfile=None, maxstep=0.1)
    opt.attach(grab, interval=1)
    opt.run(fmax=0.0, steps=_STEPS)
    return hashlib.sha256(b"".join(frames)).hexdigest()


def lj38_hashes(workers: int) -> list[str]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda _i: trajectory_sha256("lj38"), range(workers)))


def pt7_hashes(workers: int) -> list[str]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda _i: trajectory_sha256("pt7"), range(workers)))


def _diagonal(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Pure Python potential. No NumPy ufunc, so the hash is rgmin's."""
    coords = [float(value) for value in x]
    total = 0.0
    gradient: list[float] = []
    for scale, coord in zip(_SCALES, coords, strict=True):
        total += 0.5 * scale * coord * coord
        gradient.append(scale * coord)
    return total, np.array(gradient, dtype=np.float64)


def _child_env(pin: str | None, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    root = os.path.dirname(__file__)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("NPY_DISABLE_CPU_FEATURES", None)
    env.pop("PYTHONWARNINGS", None)
    if pin:
        env["NPY_DISABLE_CPU_FEATURES"] = pin
    env.update(extra)
    return env


def _run_child(code: str, *, pin: str | None, **extra: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=_child_env(pin, **extra),
        cwd=os.path.dirname(__file__),
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"child exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _pt7_expect() -> str:
    if sys.platform == "win32":
        return PT7_EMT_WINDOWS
    return PT7_EMT


def _numpy_pin() -> str | None:
    """Dispatch names for the AVX-512 and X86_V3/V4 paths, if this build has them."""
    names = json.loads(_run_child(_DISPATCH_PROBE, pin=None)[-1])
    blocked = [name for name in names if _blocked(name)]
    if not blocked:
        return None
    return ",".join(blocked)


# The child spawns a ThreadPoolExecutor of up to 8 workers. The marker
# keeps pytest-run-parallel from multiplying those pools.
@pytest.mark.thread_unsafe
def test_python_thread_invariance() -> None:
    """Lennard-Jones matches the golden at 1, 2 and 8 Python threads, pinned."""
    pytest.importorskip("ase")
    lines = _run_child(_LJ38_THREADS, pin=_numpy_pin())
    assert [line.split() for line in lines] == [[LJ38] * workers for workers in (1, 2, 8)]


def test_lj38_rayon_env_invariance() -> None:
    """Lennard-Jones matches the golden at every Rayon thread count, pinned."""
    pytest.importorskip("ase")
    pin = _numpy_pin()
    for threads in ("1", "2", "3", "8"):
        lines = _run_child(
            _LJ38_ONCE,
            pin=pin,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        )
        assert lines == [LJ38]


def test_pt7_emt_golden_pins_numpy_not_rgmin() -> None:
    """Hash Pt7 EMT with the AVX-512 and X86_V3/V4 paths disabled.

    The variable is set on the child process, before NumPy is imported.
    A conftest assignment is too late once a plugin has imported NumPy.
    Only names from this build's ``__cpu_dispatch__`` are listed, and
    the child asserts ``show_runtime`` and ``__cpu_features__``. This
    pins ASE's EMT, which is NumPy-dispatched, and not rgmin.
    Windows NumPy takes a different value. The thread counts still agree.
    A mismatch reports the observed digest with the platform and the
    NumPy version. ``RGMIN_PT7_DIGEST_OUT`` names a file for that record.
    """
    pytest.importorskip("ase")
    pin = _numpy_pin()
    expect = _pt7_expect()
    observed = {f"python_threads_{w}": h for w, h in zip((1, 2, 8), _run_child(_PT7_THREADS, pin=pin))}
    for threads in ("1", "2", "3", "8"):
        for digest in _run_child(
            _PT7_ONCE,
            pin=pin,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        ):
            observed[f"rayon_{threads}"] = digest
    record = _pt7_record(observed, expect, pin)
    out = os.environ.get("RGMIN_PT7_DIGEST_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    report = json.dumps(record, indent=2)
    assert len(observed) == 7, report
    assert set(observed.values()) == {expect}, report


def _pt7_record(observed: dict[str, str], expect: str, pin: str | None) -> dict[str, object]:
    """Everything needed to pin the Pt7 golden from one failing run."""
    import platform

    import ase

    digests = sorted(set(observed.values()))
    return {
        "observed": digests[0] if len(digests) == 1 else digests,
        "expected": expect,
        "constant": "PT7_EMT_WINDOWS" if sys.platform == "win32" else "PT7_EMT",
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "ase": ase.__version__,
        "pin": pin,
        "runs": observed,
    }


def test_rgmin_quadratic_golden() -> None:
    """rgmin's own potential matches on every runner with no kernel pin.

    Eight L-BFGS steps on a diagonal quadratic. The objective uses
    Python arithmetic. The start frame is part of the hash. The
    solver takes the default per-atom cap.
    """
    import rgmin

    x = np.array(_QUADRATIC_X0, dtype=np.float64)
    solver = rgmin.Solver(x.size, rgmin.SolverConfig("lbfgs", memory=6, gtol=0.0, maxiter=1))
    frames = [x.tobytes()]
    for _ in range(8):
        solver.step(_diagonal, x)
        frames.append(x.tobytes())
    assert hashlib.sha256(b"".join(frames)).hexdigest() == RGMIN_QUADRATIC
