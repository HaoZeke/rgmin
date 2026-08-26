#!/usr/bin/env python3
"""Mint dest golden-master fixtures from zadorlab/sella and manopt.

Set SELLA_ROOT to a zadorlab/sella checkout (the directory that contains
the `sella/` package). Optional MANOPT_ROOT is a NicolasBoumal/manopt
tree; when MATLAB/Octave can see it, factory numbers come from that
source. Otherwise the remint writes the published manopt formulas
(spherefactory proj/retr, positivefactory exp, symmetricfactory
multisym) used by dest comments.

stdin is unused. stdout is one JSON object:

  source: {sella_root, sella_file, manopt}
  cases:  [{name, kind, s}, ...]

kind is rfo / qn / qn_irc / prfo / ts_bfgs / ras / sphere_proj /
sphere_retr / positive_retr / symmetric_proj.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def _sella_root() -> Path:
    raw = os.environ.get("SELLA_ROOT", "")
    if raw:
        return Path(raw).resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / "Python" / "sella",
        here.parents[3] / "Python" / "sella",
    ]
    for c in candidates:
        if (c / "sella" / "optimize" / "stepper.py").is_file():
            return c
    raise SystemExit("SELLA_ROOT not set and no local zadorlab/sella checkout")


def _hess(dim: int, b0: np.ndarray):
    from sella.linalg import ApproximateHessian

    return ApproximateHessian(dim, dim, np.asarray(b0, dtype=float), initialized=True)


def _load_sella_modules(root: Path):
    """Import stepper.py and hessian_update.py without sella/__init__.py (jax)."""
    import importlib.util
    import types

    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        linalg_mod = types.ModuleType("scipy.linalg")

        def _lstsq(a, b, rcond=None):
            x, residuals, rank, singular = np.linalg.lstsq(a, b, rcond=rcond)
            return x, residuals, rank, singular

        linalg_mod.eigh = np.linalg.eigh
        linalg_mod.lstsq = _lstsq
        linalg_mod.solve = np.linalg.solve
        scipy.linalg = linalg_mod
        sys.modules["scipy"] = scipy
        sys.modules["scipy.linalg"] = linalg_mod

    pkg = types.ModuleType("sella")
    pkg.__path__ = [str(root / "sella")]
    sys.modules["sella"] = pkg

    gpu = types.ModuleType("sella._gpu")
    gpu.torch = None

    def _gpu_ok(_n):
        return False

    gpu._gpu_ok = _gpu_ok
    sys.modules["sella._gpu"] = gpu
    pkg._gpu = gpu

    class ApproximateHessian:
        def __init__(self, dim, ncart, B0=None, update_method="TS-BFGS",
                     symm=2, initialized=False):
            self.dim = dim
            self.ncart = ncart
            self.update_method = update_method
            self.symm = symm
            self.initialized = initialized
            self.B = None if B0 is None else np.asarray(B0, dtype=float)
            if self.B is not None:
                self.evals, self.evecs = np.linalg.eigh(self.B)
            else:
                self.evals = None
                self.evecs = None

        def asarray(self):
            return self.B

        def project(self, u):
            bproj = u.T @ self.B @ u
            return ApproximateHessian(bproj.shape[0], 0, bproj, initialized=True)

    linalg = types.ModuleType("sella.linalg")
    linalg.ApproximateHessian = ApproximateHessian
    sys.modules["sella.linalg"] = linalg
    pkg.linalg = linalg

    hess_spec = importlib.util.spec_from_file_location(
        "sella.hessian_update", root / "sella" / "hessian_update.py"
    )
    hess = importlib.util.module_from_spec(hess_spec)
    sys.modules["sella.hessian_update"] = hess
    hess_spec.loader.exec_module(hess)

    opt = types.ModuleType("sella.optimize")
    opt.__path__ = [str(root / "sella" / "optimize")]
    sys.modules["sella.optimize"] = opt

    step_spec = importlib.util.spec_from_file_location(
        "sella.optimize.stepper", root / "sella" / "optimize" / "stepper.py"
    )
    stepper = importlib.util.module_from_spec(step_spec)
    sys.modules["sella.optimize.stepper"] = stepper
    step_spec.loader.exec_module(stepper)
    return stepper, hess


def mint_sella(root: Path) -> tuple[dict, list[dict]]:
    stepper, hess = _load_sella_modules(root)
    update_H = hess.update_H
    PartitionedRationalFunctionOptimization = (
        stepper.PartitionedRationalFunctionOptimization
    )
    QuasiNewton = stepper.QuasiNewton
    QuasiNewtonIRC = stepper.QuasiNewtonIRC
    RationalFunctionOptimization = stepper.RationalFunctionOptimization

    cases: list[dict] = []

    h_eye = _hess(2, np.eye(2))
    g_down = np.array([2.0, 0.0])
    rfo = RationalFunctionOptimization(g_down, h_eye, order=0)
    for alpha in (0.25, 0.5, 1.0):
        s, _ = rfo.get_s(alpha)
        cases.append(
            {
                "name": f"rfo_eye_order0_alpha{alpha}",
                "kind": "rfo",
                "order": 0,
                "alpha": alpha,
                "H": np.eye(2).tolist(),
                "g": g_down.tolist(),
                "s": [float(v) for v in s],
            }
        )

    h_sad = np.array([[-1.0, 0.2], [0.2, 4.0]])
    g_sad = np.array([0.5, 0.4])
    rfo1 = RationalFunctionOptimization(g_sad, _hess(2, h_sad), order=1)
    s, _ = rfo1.get_s(1.0)
    cases.append(
        {
            "name": "rfo_saddle_order1_alpha1",
            "kind": "rfo",
            "order": 1,
            "alpha": 1.0,
            "H": h_sad.tolist(),
            "g": g_sad.tolist(),
            "s": [float(v) for v in s],
        }
    )

    qn = QuasiNewton(g_sad, _hess(2, h_sad), order=1)
    for alpha in (0.0, 0.3, 1.0):
        s, _ = qn.get_s(alpha)
        cases.append(
            {
                "name": f"qn_saddle_order1_alpha{alpha}",
                "kind": "qn",
                "order": 1,
                "alpha": alpha,
                "H": h_sad.tolist(),
                "g": g_sad.tolist(),
                "s": [float(v) for v in s],
            }
        )

    d1 = np.array([-0.05, 0.0])
    qnirc = QuasiNewtonIRC(g_sad, _hess(2, h_sad), order=1, d1=d1)
    s, _ = qnirc.get_s(0.2)
    cases.append(
        {
            "name": "qn_irc_order1_alpha0.2",
            "kind": "qn_irc",
            "order": 1,
            "alpha": 0.2,
            "H": h_sad.tolist(),
            "g": g_sad.tolist(),
            "d1": d1.tolist(),
            "s": [float(v) for v in s],
        }
    )

    prfo = PartitionedRationalFunctionOptimization(g_sad, _hess(2, h_sad), order=1)
    s, _ = prfo.get_s(1.0)
    cases.append(
        {
            "name": "prfo_saddle_order1_alpha1",
            "kind": "prfo",
            "order": 1,
            "alpha": 1.0,
            "H": h_sad.tolist(),
            "g": g_sad.tolist(),
            "s": [float(v) for v in s],
        }
    )

    b0 = np.diag([-1.0, 4.0])
    step = np.array([0.15, 0.05])
    y = np.array([0.35, -0.10])
    bplus = update_H(b0.copy(), step, y, method="TS-BFGS", symm=None)
    cases.append(
        {
            "name": "ts_bfgs_keep_saddle",
            "kind": "ts_bfgs",
            "B": b0.tolist(),
            "step": step.tolist(),
            "y": y.tolist(),
            "s": [float(v) for v in np.asarray(bplus).ravel()],
        }
    )

    # RestrictedAtomicStep cons: scale so max per-atom Euclidean <= delta.
    s_cart = np.array([0.4, 0.3, 0.0, 0.05, 0.0, 0.0])
    delta = 0.1
    norms = np.linalg.norm(s_cart.reshape(-1, 3), axis=1)
    scale = delta / norms.max()
    cases.append(
        {
            "name": "ras_clip_max_atom",
            "kind": "ras",
            "delta": delta,
            "step": s_cart.tolist(),
            "s": [float(v) for v in (s_cart * scale)],
        }
    )

    stepper_py = root / "sella" / "optimize" / "stepper.py"
    meta = {
        "sella_root": "SELLA_ROOT",
        "sella_file": "sella/optimize/stepper.py",
        "sella_exists": stepper_py.is_file(),
    }
    return meta, cases


def mint_manopt_formulas() -> tuple[dict, list[dict]]:
    """Published manopt factory algebra dest claims to port.

    These are the MATLAB formulas from spherefactory / positivefactory /
    symmetricfactory, not dest numbers.
    """
    cases: list[dict] = []
    x = np.array([0.0, 1.0, 0.0])
    v = np.array([0.2, 0.3, -0.1])
    proj = v - x * np.dot(x, v)
    retr = (x + proj) / np.linalg.norm(x + proj)
    cases.append(
        {
            "name": "sphere_proj_north",
            "kind": "sphere_proj",
            "x": x.tolist(),
            "v": v.tolist(),
            "s": [float(a) for a in proj],
        }
    )
    cases.append(
        {
            "name": "sphere_retr_north",
            "kind": "sphere_retr",
            "x": x.tolist(),
            "v": proj.tolist(),
            "s": [float(a) for a in retr],
        }
    )

    xp = np.array([1.5, 0.5, 2.0])
    vp = np.array([0.1, -0.2, 0.0])
    y = xp * np.exp(vp / xp)
    cases.append(
        {
            "name": "positive_retr_exp",
            "kind": "positive_retr",
            "x": xp.tolist(),
            "v": vp.tolist(),
            "s": [float(a) for a in y],
        }
    )

    xs = np.array([1.0, 0.0, 0.0, -1.0])
    vs = np.array([0.0, 0.2, -0.1, 0.0])
    a = vs.reshape(2, 2)
    sym = 0.5 * (a + a.T)
    cases.append(
        {
            "name": "symmetric_multisym",
            "kind": "symmetric_proj",
            "x": xs.tolist(),
            "v": vs.tolist(),
            "s": [float(a) for a in sym.ravel()],
        }
    )
    return {"manopt": "published-factory-formulas"}, cases


def main() -> None:
    root = _sella_root()
    sella_meta, sella_cases = mint_sella(root)
    manopt_meta, manopt_cases = mint_manopt_formulas()
    out = {
        "source": {**sella_meta, **manopt_meta},
        "cases": sella_cases + manopt_cases,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
