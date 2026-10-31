"""Build the C minimize library and the asv helper beside it.

Develop has no Python package. asv's build step compiles the shared
library with meson, then links benchmarks/asv_quadratic.c against it.
The helper is build/librgmin_asv.so. The benchmark loads that file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _need(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise SystemExit(f"{name} was not found on PATH")
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd()
    meson = _need("meson")
    _need("ninja")
    gcc = _need("gcc")
    gxx = _need("g++")
    env = os.environ.copy()
    # Meson probes c++. A clang driver that does not search the gcc
    # lib directory fails to link libstdc++. gcc and g++ do search it.
    env["CC"] = gcc
    env["CXX"] = gxx
    subprocess.check_call(
        [meson, "setup", "build", "-Drgmin_features=capi"],
        cwd=root,
        env=env,
    )
    subprocess.check_call(
        [meson, "compile", "-C", "build", "rgmin-cargo"],
        cwd=root,
        env=env,
    )
    helper = root / "benchmarks" / "asv_quadratic.c"
    if not helper.is_file():
        raise SystemExit(f"missing {helper}")
    subprocess.check_call(
        [
            gcc,
            "-shared",
            "-fPIC",
            "-O2",
            "-Iinclude",
            str(helper),
            "-Lbuild",
            "-l:librgmin.so",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            "build/librgmin_asv.so",
        ],
        cwd=root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
