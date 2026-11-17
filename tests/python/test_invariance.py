# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""SHA-256 of trajectories at several thread counts.

LJ38 and Pt7 thread and Rayon invariance compare every run with one
reference run on the same machine. The LJ38 and Pt7 goldens pin ASE's
LennardJones and EMT, which are NumPy-dispatched and call the platform
libm. They are not pins of rgmin. Both digests are pinned on glibc
only. Every platform compares the frames and energies with the golden
run within a tolerance. rgmin's own results must be bit-identical
across ISAs with no environment variable. That check is the diagonal
quadratic.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

# One hash per fixture. 3.14 and 3.14t both compare against these values.
# ASE LennardJones on glibc, in a process started with the pin below.
# ``(sigma**2 / r2) ** 3`` calls libm ``pow`` for every atom and each of
# its neighbours. In the third force call atoms 4 and 17 are 0.383
# apart, and the cube of 1 / r**2 = LJ38_PROBE lies 0.0013 ulp above the
# midpoint between two doubles. glibc returns the upper one on x86_64
# and aarch64. NumPy's AVX512F power kernel returns the lower one, and
# macOS arm64 gives the digest of exactly that rounding, e3f64018...
# The last three frames move by at most 3e-17. The digest pins glibc's
# pow and not rgmin.
LJ38 = "9ff5ac4e272b58e733c2336e8b2580a566cfd0eefd77c3c1797359fb024cf24a"
LJ38_PROBE = float.fromhex("0x1.b499f0a26aea2p+2")
LJ38_PROBE_CUBE = 317.4791692719938
# Machines where glibc gives LJ38 and PT7_EMT. The pin on both ISAs
# also sees a rounding change that only one of them takes, such as a
# fused multiply-add on aarch64.
GLIBC_MACHINES = frozenset({"x86_64", "aarch64"})
_ON_GLIBC = platform.libc_ver()[0] == "glibc" and platform.machine() in GLIBC_MACHINES
# Frames and energies of the LJ38 run, recorded on Linux x86_64 under
# the pin. The SHA-256 of the frames is LJ38.
_LJ38_GOLDEN = os.path.join(os.path.dirname(__file__), "data", "lj38_golden.json")
# Rounding the probe cube the other way moves LJ38 positions by at most
# 3e-17 and energies by 2e-16 relative. Moving every cube one ulp up or
# down at random moves them by at most 9e-16 and 1.4e-15 over 50 seeds.
# The bounds sit about three decades above that and ten below a step.
_LJ38_POSITION_ATOL = 1e-12
_LJ38_ENERGY_RTOL = 1e-12
# ASE EMT on glibc, in a process started with the pin below. Without
# AVX-512, NumPy's float64 exp calls libm ``exp`` for each element. In
# the sixth force call atoms 0 and 1 are 2.348 apart, and
# ``exp(-eta2 * (r - beta * s0))`` in sigma1 has the argument PT7_PROBE.
# Its exponential lies 0.0004 ulp above the midpoint between two
# doubles. glibc returns the upper one on x86_64 and aarch64. Apple's
# x86_64 libm returns the lower one, and macOS x86_64 gives the digest
# of exactly that rounding, 61fdafb0... The last frame moves by at most
# 4.4e-16. The digest pins glibc's exp and not rgmin.
PT7_EMT = "299f8164ab0df6f3b59731098b3d83e12c13c0b4117426437bf9af9df0a17ce8"
PT7_PROBE = float.fromhex("0x1.7742e155039b9p+0")
PT7_PROBE_EXP = 4.331284983478651
# Frames and energies of the Pt7 run, recorded on Linux x86_64 under
# the pin. The SHA-256 of the frames is PT7_EMT.
_PT7_GOLDEN = os.path.join(os.path.dirname(__file__), "data", "pt7_golden.json")
# Rounding exp(PT7_PROBE) the other way moves Pt7 positions by at most
# 4.4e-16 and energies by 7e-16 relative. Moving every exp and log one
# ulp up, one down, or not at all at random moves them by at most
# 2.1e-14 and 9.8e-15 over 200 seeds. The bounds sit fifty times above
# that and eleven decades below a step.
_PT7_POSITION_ATOL = 1e-12
_PT7_ENERGY_RTOL = 1e-12
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

_PT7_RUN = """
import json
import numpy as np
from test_invariance import PT7_PROBE, assert_numpy_dispatch_disabled, trajectory_frames
assert_numpy_dispatch_disabled()
record = trajectory_frames("pt7")
record["exp"] = float(np.exp(np.array([PT7_PROBE]))[0])
print(json.dumps(record))
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

_LJ38_RUN = """
import json
import numpy as np
from test_invariance import LJ38_PROBE, assert_numpy_dispatch_disabled, trajectory_frames
assert_numpy_dispatch_disabled()
record = trajectory_frames("lj38")
record["cube"] = float((np.array([LJ38_PROBE]) ** 3)[0])
print(json.dumps(record))
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


def _trajectory(kind: str, energies: list[float] | None = None) -> list[np.ndarray]:
    """Positions at every frame of the five-step run. Energies only on request."""
    from rgmin.ase import LBFGS

    atoms = _lj38() if kind == "lj38" else _pt7()
    frames: list[np.ndarray] = []

    def grab() -> None:
        frames.append(np.array(atoms.get_positions(), dtype=np.float64))
        if energies is not None:
            energies.append(float(atoms.get_potential_energy()))

    opt = LBFGS(atoms, logfile=None, maxstep=0.1)
    opt.attach(grab, interval=1)
    opt.run(fmax=0.0, steps=_STEPS)
    return frames


def _frames_sha256(frames: Any) -> str:
    return hashlib.sha256(np.asarray(frames, dtype=np.float64).tobytes()).hexdigest()


def trajectory_sha256(kind: str) -> str:
    return _frames_sha256(_trajectory(kind))


def trajectory_frames(kind: str) -> dict[str, object]:
    """Energies and positions at every frame, as lists for JSON."""
    energies: list[float] = []
    frames = _trajectory(kind, energies)
    return {"energies": energies, "positions": [frame.tolist() for frame in frames]}


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


def _numpy_pin() -> str | None:
    """Dispatch names for the AVX-512 and X86_V3/V4 paths, if this build has them."""
    names = json.loads(_run_child(_DISPATCH_PROBE, pin=None)[-1])
    blocked = [name for name in names if _blocked(name)]
    if not blocked:
        return None
    return ",".join(blocked)


@pytest.fixture(scope="module")
def lj38_run() -> dict[str, Any]:
    """One pinned Lennard-Jones run on this machine, the reference for the LJ38 tests.

    Frames, energies, their digest, and how this libm rounds the cube
    of LJ38_PROBE.
    """
    pytest.importorskip("ase")
    run: dict[str, Any] = json.loads(_run_child(_LJ38_RUN, pin=_numpy_pin())[-1])
    run["digest"] = _frames_sha256(run["positions"])
    return run


@pytest.fixture(scope="module")
def pt7_run() -> dict[str, Any]:
    """One pinned EMT run on this machine, the reference for the Pt7 tests.

    Frames, energies, their digest, and how this libm rounds the
    exponential of PT7_PROBE.
    """
    pytest.importorskip("ase")
    run: dict[str, Any] = json.loads(_run_child(_PT7_RUN, pin=_numpy_pin())[-1])
    run["digest"] = _frames_sha256(run["positions"])
    return run


def _assert_golden_trajectory(
    run: dict[str, Any], path: str, digest: str, position_atol: float, energy_rtol: float
) -> None:
    """Compare a pinned run with the golden frames and energies at ``path``, which hash to ``digest``."""
    with open(path, encoding="utf-8") as handle:
        golden = json.load(handle)
    assert _frames_sha256(golden["positions"]) == digest
    where = f"{sys.platform} {platform.machine()}, NumPy {np.__version__}"
    np.testing.assert_allclose(
        np.asarray(run["positions"], dtype=np.float64),
        np.asarray(golden["positions"], dtype=np.float64),
        rtol=0.0,
        atol=position_atol,
        err_msg=where,
    )
    np.testing.assert_allclose(
        np.asarray(run["energies"], dtype=np.float64),
        np.asarray(golden["energies"], dtype=np.float64),
        rtol=energy_rtol,
        atol=0.0,
        err_msg=where,
    )


# The child spawns a ThreadPoolExecutor of up to 8 workers. The marker
# keeps pytest-run-parallel from multiplying those pools.
@pytest.mark.thread_unsafe
def test_python_thread_invariance(lj38_run: dict[str, Any]) -> None:
    """Lennard-Jones matches this machine's run at 1, 2 and 8 Python threads, pinned."""
    digest = lj38_run["digest"]
    lines = _run_child(_LJ38_THREADS, pin=_numpy_pin())
    assert [line.split() for line in lines] == [[digest] * workers for workers in (1, 2, 8)]


def test_lj38_rayon_env_invariance(lj38_run: dict[str, Any]) -> None:
    """Lennard-Jones matches this machine's run at every Rayon thread count, pinned."""
    pin = _numpy_pin()
    for threads in ("1", "2", "3", "8"):
        lines = _run_child(
            _LJ38_ONCE,
            pin=pin,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        )
        assert lines == [lj38_run["digest"]], threads


@pytest.mark.skipif(
    not _ON_GLIBC,
    reason="the LJ38 digest pins glibc's pow; test_lj38_golden_trajectory covers this platform",
)
def test_lj38_golden_digest(lj38_run: dict[str, Any]) -> None:
    """This machine's Lennard-Jones run matches the golden digest on glibc.

    A mismatch with the probe cube still at LJ38_PROBE_CUBE did not come
    from that rounding.
    """
    assert lj38_run["digest"] == LJ38, (
        f"{lj38_run['digest']} on {sys.platform} {platform.machine()} with NumPy {np.__version__}: "
        f"LJ38_PROBE ** 3 = {lj38_run['cube']!r}, glibc gives {LJ38_PROBE_CUBE!r}"
    )


def test_lj38_golden_trajectory(lj38_run: dict[str, Any]) -> None:
    """This machine's Lennard-Jones frames and energies match the golden run.

    A tolerance and not a digest, because another libm may round a cube
    on the path the other way. The stored frames hash to LJ38.
    """
    _assert_golden_trajectory(lj38_run, _LJ38_GOLDEN, LJ38, _LJ38_POSITION_ATOL, _LJ38_ENERGY_RTOL)


def test_pt7_emt_thread_invariance(pt7_run: dict[str, Any]) -> None:
    """EMT matches this machine's run at 1, 2 and 8 Python threads and every Rayon thread count.

    Every run hashes Pt7 EMT with the AVX-512 and X86_V3/V4 paths
    disabled. ``NPY_DISABLE_CPU_FEATURES`` is set on the child process,
    before NumPy is imported. A conftest assignment is too late once a
    plugin has imported NumPy. Only names from this build's
    ``__cpu_dispatch__`` are listed, and the child asserts
    ``show_runtime`` and ``__cpu_features__``. A mismatch reports every
    digest with the platform and the NumPy version.
    ``RGMIN_PT7_DIGEST_OUT`` names a file for that record.
    """
    pin = _numpy_pin()
    observed = {f"python_threads_{w}": h for w, h in zip((1, 2, 8), _run_child(_PT7_THREADS, pin=pin))}
    for threads in ("1", "2", "3", "8"):
        for digest in _run_child(
            _PT7_ONCE,
            pin=pin,
            RAYON_NUM_THREADS=threads,
            OMP_NUM_THREADS=threads,
        ):
            observed[f"rayon_{threads}"] = digest
    record = _pt7_record(observed, pt7_run, pin)
    out = os.environ.get("RGMIN_PT7_DIGEST_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    report = json.dumps(record, indent=2)
    assert len(observed) == 7, report
    assert set(observed.values()) == {pt7_run["digest"]}, report


def _pt7_record(observed: dict[str, str], run: dict[str, Any], pin: str | None) -> dict[str, object]:
    """The seven Pt7 digests beside this machine's reference run and the glibc golden."""
    import ase

    digests = sorted(set(observed.values()))
    return {
        "observed": digests[0] if len(digests) == 1 else digests,
        "reference": run["digest"],
        "golden": PT7_EMT,
        "exp": run["exp"],
        "platform": sys.platform,
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "python": sys.version,
        "numpy": np.__version__,
        "ase": ase.__version__,
        "pin": pin,
        "runs": observed,
    }


@pytest.mark.skipif(
    not _ON_GLIBC,
    reason="the Pt7 digest pins glibc's exp; test_pt7_emt_golden_trajectory covers this platform",
)
def test_pt7_emt_golden_digest(pt7_run: dict[str, Any]) -> None:
    """This machine's EMT run matches the golden digest on glibc.

    A mismatch with the exponential still at PT7_PROBE_EXP did not come
    from that rounding.
    """
    assert pt7_run["digest"] == PT7_EMT, (
        f"{pt7_run['digest']} on {sys.platform} {platform.machine()} with NumPy {np.__version__}: "
        f"exp(PT7_PROBE) = {pt7_run['exp']!r}, glibc gives {PT7_PROBE_EXP!r}"
    )


def test_pt7_emt_golden_trajectory(pt7_run: dict[str, Any]) -> None:
    """This machine's EMT frames and energies match the golden run.

    A tolerance and not a digest, because another libm may round an
    exponential on the path the other way. The stored frames hash to
    PT7_EMT.
    """
    _assert_golden_trajectory(pt7_run, _PT7_GOLDEN, PT7_EMT, _PT7_POSITION_ATOL, _PT7_ENERGY_RTOL)


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
