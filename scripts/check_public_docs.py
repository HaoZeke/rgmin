#!/usr/bin/env python3
"""Fail when a public Rust item has no doc comment.

``src/ffi.rs`` is the C ABI mirror. Its contract is ``include/rgmin.h``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ITEM = re.compile(
    r"^\s*pub(?:\s*\([^)]*\))?\s+(?:async\s+)?(?:fn|struct|enum|trait|const|type|mod|use|static)\b"
)
PUBLIC = re.compile(r"^\s*pub(?:\s*\([^)]*\))?")
CRATE_ONLY = re.compile(r"^\s*pub\s*\(\s*(?:crate|super|in)\b")


def undocumented(path: Path) -> list[int]:
    pending = False
    attr = 0
    missing: list[int] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if attr:
            attr += stripped.count("[") - stripped.count("]")
            attr = max(attr, 0)
            continue
        if stripped.startswith("///") or stripped.startswith("/**") or stripped.startswith("/*!"):
            pending = True
            continue
        if not stripped or stripped.startswith("//"):
            pending = False
            continue
        if stripped.startswith("#["):
            attr += stripped.count("[") - stripped.count("]")
            attr = max(attr, 0)
            continue
        if stripped.startswith("pub use "):
            pending = False
            continue
        if ITEM.match(line) and not CRATE_ONLY.match(line) and PUBLIC.match(line):
            if not pending:
                missing.append(number)
            pending = False
            continue
        pending = False
    return missing


def main() -> int:
    failed = False
    for path in sorted(SRC.rglob("*.rs")):
        if path.name == "ffi.rs":
            continue
        lines = undocumented(path)
        for line in lines:
            failed = True
            print(f"{path.relative_to(ROOT)}:{line}: public item has no doc comment")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
