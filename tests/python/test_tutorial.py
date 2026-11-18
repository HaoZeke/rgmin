# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""The Python quickstart runs as written, and so does the example it names."""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "docs" / "orgmode" / "tutorials" / "python.org"
BLOCK = re.compile(r"^#\+begin_src python\n(.*?)^#\+end_src", re.S | re.M)


def test_python_tutorial_blocks_run_in_order() -> None:
    if not TUTORIAL.is_file():
        pytest.skip("docs are not part of this checkout")
    blocks = BLOCK.findall(TUTORIAL.read_text())
    assert len(blocks) >= 2
    scope: dict[str, object] = {}
    for source in blocks:
        exec(compile(source, str(TUTORIAL), "exec"), scope)


def test_minimize_quadratic_example_runs() -> None:
    example = ROOT / "examples" / "minimize_quadratic.py"
    if not example.is_file():
        pytest.skip("examples are not part of this checkout")
    runpy.run_path(str(example), run_name="__main__")
