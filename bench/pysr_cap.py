# Copyright (c) rgmin developers
"""Fit the eOn reset bit and the step length from a step log.

Train on lj75, score on lj150. The features are the taken step, not the
proposal, so a low error describes that step. It is not installed as a
reset rule.

    python bench/pysr_cap.py bench/raw/steps.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from pysr import PySRRegressor

FEATURES = [
    "curvature",
    "fnorm",
    "fnorm_prev",
    "cos_angle",
    "cap_ratio",
    "atom_mean",
    "atom_max",
]


def fit(x, y):
    model = PySRRegressor(
        niterations=8,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        maxsize=12,
        populations=8,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )
    model.fit(x, y)
    return model


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/raw/steps.csv"
    df = pd.read_csv(path)
    train = df[df.problem == "lj75"]
    test = df[df.problem == "lj150"]
    xtr = train[FEATURES].to_numpy()
    xte = test[FEATURES].to_numpy()
    reset = fit(xtr, train.reset.to_numpy(dtype=float))
    length = fit(xtr, train.step_len.to_numpy(dtype=float))
    rhat = reset.predict(xte)
    bit = (rhat >= 0.5).astype(float)
    y = test.reset.to_numpy(dtype=float)
    acc = float(np.mean(bit == y))
    # A step taken along the force is what a reset writes.
    base = (test.cos_angle.to_numpy() > 0.999).astype(float)
    base_acc = float(np.mean(base == y))
    lhat = length.predict(xte)
    ylen = test.step_len.to_numpy(dtype=float)
    mae = float(np.mean(np.abs(lhat - ylen)))
    mean_mae = float(np.mean(np.abs(ylen - ylen.mean())))
    print("reset", reset.get_best().equation, "heldout_acc", acc, "parallel_acc", base_acc)
    print("length", length.get_best().equation, "heldout_mae", mae, "mean_mae", mean_mae)


if __name__ == "__main__":
    main()
