#!/usr/bin/env python3
"""Fail when a function declared in ``include/rgmin.h`` has no doc comment.

``include/rgmin.h`` is the C ABI contract, and Doxygen builds the C
reference from it. A declaration counts as documented when a ``/** */``
or ``///`` comment ends on the line above it. A comment above an enum
typedef documents the enum and not the setter declared after it, so that
setter needs a comment of its own.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "rgmin.h"
DECL = re.compile(r"^[A-Za-z_][\w\s\*]*?\b(rgmin_\w+)\s*\(")
SKIP = ("#", "typedef", "extern", "_Static_assert", "static_assert", "}")


def undocumented(lines: list[str]) -> list[tuple[int, str]]:
    missing: list[tuple[int, str]] = []
    in_comment = False
    previous_ends_comment = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if in_comment:
            if "*/" in stripped:
                in_comment = False
                previous_ends_comment = True
            continue
        if stripped.startswith("/*"):
            if "*/" in stripped:
                previous_ends_comment = stripped.startswith("/**")
            else:
                in_comment = True
            continue
        if stripped.startswith("///"):
            previous_ends_comment = True
            continue
        if not stripped:
            previous_ends_comment = False
            continue
        match = DECL.match(line)
        if match and not stripped.startswith(SKIP) and not previous_ends_comment:
            missing.append((number, match.group(1)))
        previous_ends_comment = False
    return missing


def main() -> int:
    lines = HEADER.read_text().splitlines()
    missing = undocumented(lines)
    for number, name in missing:
        print(f"include/rgmin.h:{number}: {name} has no doc comment")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
