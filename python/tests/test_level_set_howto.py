# Copyright (c) 2026 Rohit Goswami
# SPDX-License-Identifier: MIT
"""The level-set how-to snippet runs as written."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

HOWTO = Path(__file__).resolve().parents[2] / "docs" / "orgmode" / "howto" / "level-set.org"
BLOCK = re.compile(r"^#\+begin_src python\n(.*?)^#\+end_src", re.S | re.M)


def test_level_set_howto_block_runs() -> None:
    if not HOWTO.is_file():
        pytest.skip("docs are not part of this checkout")
    blocks = BLOCK.findall(HOWTO.read_text())
    assert len(blocks) == 1
    exec(compile(blocks[0], str(HOWTO), "exec"), {})
