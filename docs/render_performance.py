#!/usr/bin/env python3
"""Write the performance page from the committed benchmark tables."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "bench" / "raw"
OUT = ROOT / "docs" / "source" / "explanation" / "performance.rst"

TABLES = (
    ("rgmin.csv", "rgmin"),
    ("eon.csv", "eOn"),
    ("ase.csv", "ASE"),
    ("scipy.csv", "SciPy"),
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def table(title: str, records: list[dict[str, str]]) -> list[str]:
    picked = [
        row
        for row in records
        if row["problem"] in {"Rosenbrock2", "Rosenbrock10", "LJ7"}
        and row["converged"] == "yes"
    ]
    lines = [
        title,
        "-" * len(title),
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Problem",
        "     - Arm",
        "     - Evaluations",
        "     - Gradient",
    ]
    for row in picked:
        lines.extend(
            [
                f"   * - {row['problem']}",
                f"     - {row['arm']}",
                f"     - {row['evals']}",
                f"     - {row['grad']}",
            ]
        )
    lines.append("")
    return lines


def main() -> None:
    body = [
        "Performance",
        "===========",
        "",
        "Generated from ``bench/raw``. Do not edit the table by hand.",
        "``docs/render_performance.py`` rewrites this page.",
        "",
        "The eOn L-BFGS arm and ``rgmin-eon`` spend the same evaluations",
        "on Rosenbrock-2. The scaled arm is ASE's length control. On",
        "Rosenbrock-10 it stops in the well near ``x0 = -1``; the cap does not.",
        "That comparison is ``tests/scaled_rosenbrock10.rs``.",
        "",
    ]
    for name, title in TABLES:
        body.extend(table(title, rows(RAW / name)))
    OUT.write_text("\n".join(body) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
