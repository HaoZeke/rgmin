#!/usr/bin/env python3
"""Plot the largest per-atom force against force calls for relax_morse.

    python3 docs/figures/relax-morse/plot_convergence.py [trace.csv]

Without an argument the script runs
``cargo run --release --example relax_morse`` from the repository root
and plots its stdout. Needs matplotlib.
"""

import csv
import io
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = {"lbfgs": "L-BFGS (m = 20)", "fire": "FIRE", "fire2": "FIRE 2.0"}
FMAX = 0.01


def trace():
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1]).read_text()
    cmd = ["cargo", "run", "-q", "--release", "--example", "relax_morse"]
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def main():
    runs = {}
    for row in csv.DictReader(io.StringIO(trace())):
        calls, fmax = runs.setdefault(row["method"], ([], []))
        calls.append(int(row["force_calls"]))
        fmax.append(float(row["fmax"]))

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    for name, (calls, fmax) in runs.items():
        ax.semilogy(calls, fmax, marker="o", markersize=3, label=LABELS.get(name, name))
    ax.axhline(FMAX, color="grey", linestyle="--", linewidth=1)
    ax.annotate("stop: 0.01 eV/Å", (0, FMAX), xytext=(2, 4), textcoords="offset points",
                color="grey", fontsize=8)
    ax.set_xlabel("force calls")
    ax.set_ylabel("largest per-atom force (eV/Å)")
    ax.set_title("Pt256 Morse, tests/data/morse_pos_0000.con", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "convergence.png", metadata={"Software": None})


if __name__ == "__main__":
    main()
