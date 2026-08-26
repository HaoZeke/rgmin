#!/usr/bin/env python3
"""Mint dest golden-master fixtures from zadorlab/sella and manopt.

Set SELLA_ROOT to a zadorlab/sella checkout (the directory that contains
the `sella/` package). Optional MANOPT_ROOT is a NicolasBoumal/manopt
tree; when MATLAB/Octave can see it, factory numbers come from that
source. Otherwise the remint writes the published manopt formulas
(sphere / positive / symmetric / complexcircle / multinomial /
centeredmatrix / sympositivedefinite proj, retr, transp) used by
dest comments.

stdin is unused. stdout is one JSON object:

  source: {sella_root, sella_file, manopt, manopt_files?}
  cases:  [{name, kind, s}, ...]

kind is rfo / qn / qn_irc / prfo / ts_bfgs / ras plus factory
proj / retr / transp for Sphere, Symmetric, Positive,
ComplexCircle, Multinomial, CenteredMatrix, and SPD.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# NicolasBoumal/manopt factory sources dest comments cite.
_MANOPT_FACTORY_RELS = (
    "manopt/manifolds/sphere/spherefactory.m",
    "manopt/manifolds/positive/positivefactory.m",
    "manopt/manifolds/euclidean/symmetricfactory.m",
    "manopt/manifolds/complexcircle/complexcirclefactory.m",
    "manopt/manifolds/multinomial/multinomialfactory.m",
    "manopt/manifolds/euclidean/centeredmatrixfactory.m",
    "manopt/manifolds/symfixedrank/sympositivedefinitefactory.m",
)


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


def _manopt_root() -> Path | None:
    raw = os.environ.get("MANOPT_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _manopt_clone_root(root: Path) -> Path:
    """Accept the NicolasBoumal/manopt clone or its inner manopt/ directory."""
    if (root / "manopt" / "manifolds" / "sphere" / "spherefactory.m").is_file():
        return root
    if (root / "manifolds" / "sphere" / "spherefactory.m").is_file():
        return root.parent if root.name == "manopt" else root
    raise SystemExit(f"MANOPT_ROOT={root} is not a NicolasBoumal/manopt tree")


def _locate_manopt_factories(root: Path) -> list[Path]:
    """Open the seven dest factory files under MANOPT_ROOT."""
    clone = _manopt_clone_root(root)
    found: list[Path] = []
    for rel in _MANOPT_FACTORY_RELS:
        path = clone / rel
        if not path.is_file() and rel.startswith("manopt/"):
            path = clone / rel[len("manopt/") :]
        if not path.is_file():
            hits = sorted(root.rglob(Path(rel).name))
            if hits:
                path = hits[0]
        if not path.is_file():
            raise SystemExit(f"MANOPT_ROOT={root} missing {Path(rel).name}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if "function M =" not in text:
            raise SystemExit(f"{path} is not a manopt factory")
        found.append(path)
    return found


def _rel_manopt_file(root: Path, path: Path) -> str:
    clone = _manopt_clone_root(root)
    resolved = path.resolve()
    for base in (clone, root, clone / "manopt"):
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def _floats(a) -> list[float]:
    return [float(v) for v in np.asarray(a, dtype=float).ravel(order="C")]


def _factory_case(name: str, kind: str, x, v, s, **extra) -> dict:
    out = {
        "name": name,
        "kind": kind,
        "x": _floats(x),
        "v": _floats(v),
        "s": _floats(s),
    }
    out.update(extra)
    return out


def _cplx_from_interleaved(reim) -> np.ndarray:
    a = np.asarray(reim, dtype=float).ravel()
    return a[0::2] + 1j * a[1::2]


def _cplx_to_interleaved(z) -> np.ndarray:
    z = np.asarray(z, dtype=complex).ravel()
    out = np.empty(2 * z.size, dtype=float)
    out[0::2] = z.real
    out[1::2] = z.imag
    return out


def _multisym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _center_cols(a: np.ndarray) -> np.ndarray:
    return a - a.mean(axis=1, keepdims=True)


def _published_factory_cases() -> list[dict]:
    """Published manopt factory algebra dest claims to port.

    These are the MATLAB formulas from spherefactory, positivefactory,
    symmetricfactory, complexcirclefactory, multinomialfactory,
    centeredmatrixfactory, and sympositivedefinitefactory, not dest
    numbers. Each named dest kind has proj, retr, and transp.
    """
    cases: list[dict] = []

    # spherefactory: proj d - x*(x'*d), retr (x+d)/norm, transp proj at arrival.
    x = np.array([0.0, 1.0, 0.0])
    v = np.array([0.2, 0.3, -0.1])
    proj = v - x * np.dot(x, v)
    retr = (x + proj) / np.linalg.norm(x + proj)
    transp = proj - retr * np.dot(retr, proj)
    cases.append(_factory_case("sphere_proj_north", "sphere_proj", x, v, proj))
    cases.append(_factory_case("sphere_retr_north", "sphere_retr", x, proj, retr))
    cases.append(
        _factory_case(
            "sphere_transp_north",
            "sphere_transp",
            x,
            proj,
            transp,
            x_to=_floats(retr),
        )
    )

    # positivefactory: proj identity, retr X.*exp(eta./X), transp identity.
    xp = np.array([1.5, 0.5, 2.0])
    vp = np.array([0.1, -0.2, 0.0])
    y = xp * np.exp(vp / xp)
    cases.append(_factory_case("positive_proj_id", "positive_proj", xp, vp, vp))
    cases.append(_factory_case("positive_retr_exp", "positive_retr", xp, vp, y))
    cases.append(
        _factory_case(
            "positive_transp_id",
            "positive_transp",
            xp,
            vp,
            vp,
            x_to=_floats(y),
        )
    )

    # symmetricfactory: proj multisym, retr X+U, transp identity.
    xs = np.array([[1.0, 0.0], [0.0, -1.0]])
    vs = np.array([[0.0, 0.2], [-0.1, 0.0]])
    sym = _multisym(vs)
    eta_s = np.array([[0.0, 0.05], [0.05, 0.0]])
    retr_s = xs + eta_s
    cases.append(_factory_case("symmetric_multisym", "symmetric_proj", xs, vs, sym))
    cases.append(_factory_case("symmetric_retr_add", "symmetric_retr", xs, eta_s, retr_s))
    cases.append(
        _factory_case(
            "symmetric_transp_id",
            "symmetric_transp",
            xs,
            eta_s,
            eta_s,
            x_to=_floats(retr_s),
        )
    )

    # complexcirclefactory: dest interleaved (re, im).
    # proj u - real(conj(u).*z).*z; retr sign(z+v); transp proj at arrival.
    zc = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    uc = np.array([0.5 + 0.25j, -0.1 + 0.8j])
    xc = _cplx_to_interleaved(zc)
    vc = _cplx_to_interleaved(uc)
    pc = uc - np.real(np.conjugate(uc) * zc) * zc
    rc = (zc + pc) / np.abs(zc + pc)
    tc = pc - np.real(np.conjugate(pc) * rc) * rc
    cases.append(_factory_case("complexcircle_proj_pair", "complexcircle_proj", xc, vc, _cplx_to_interleaved(pc)))
    cases.append(
        _factory_case(
            "complexcircle_retr_sign",
            "complexcircle_retr",
            xc,
            _cplx_to_interleaved(pc),
            _cplx_to_interleaved(rc),
        )
    )
    cases.append(
        _factory_case(
            "complexcircle_transp_arrive",
            "complexcircle_transp",
            xc,
            _cplx_to_interleaved(pc),
            _cplx_to_interleaved(tc),
            x_to=_floats(_cplx_to_interleaved(rc)),
        )
    )

    # multinomialfactory m=1: proj eta-(sum eta)*x; retr X.*exp(eta./X) then
    # renormalize; transp proj at arrival.
    xm = np.array([0.2, 0.3, 0.5])
    vm = np.array([1.0, 2.0, 3.0])
    pm = vm - vm.sum() * xm
    eta_m = np.array([0.1, -0.05, -0.05])
    ym = xm * np.exp(eta_m / xm)
    ym = ym / ym.sum()
    ym = np.maximum(ym, np.finfo(float).eps)
    tm = eta_m - eta_m.sum() * ym
    cases.append(_factory_case("multinomial_proj_fisher", "multinomial_proj", xm, vm, pm))
    cases.append(_factory_case("multinomial_retr_exp", "multinomial_retr", xm, eta_m, ym))
    cases.append(
        _factory_case(
            "multinomial_transp_arrive",
            "multinomial_transp",
            xm,
            eta_m,
            tm,
            x_to=_floats(ym),
        )
    )

    # centeredmatrixfactory default cols: proj center, retr X+U, transp identity.
    xcm = np.array([[1.0, -1.0], [0.0, 0.0]])
    vcm = np.array([[1.0, 3.0], [2.0, 4.0]])
    pcm = _center_cols(vcm)
    eta_c = _center_cols(np.array([[0.2, -0.2], [-0.1, 0.1]]))
    ycm = xcm + eta_c
    cases.append(_factory_case("centered_proj_cols", "centered_proj", xcm, vcm, pcm))
    cases.append(_factory_case("centered_retr_add", "centered_retr", xcm, eta_c, ycm))
    cases.append(
        _factory_case(
            "centered_transp_id",
            "centered_transp",
            xcm,
            eta_c,
            eta_c,
            x_to=_floats(ycm),
        )
    )

    # sympositivedefinitefactory: proj symm, retr symm(X+U+0.5 U X^{-1} U),
    # default transp identity.
    xspd = np.eye(2)
    vspd = np.array([[0.3, 1.0], [-0.4, 0.5]])
    pspd = _multisym(vspd)
    eta_p = np.array([[0.0, 0.2], [0.2, 0.0]])
    yspd = _multisym(xspd + eta_p + 0.5 * eta_p @ np.linalg.solve(xspd, eta_p))
    cases.append(_factory_case("spd_proj_symm", "spd_proj", xspd, vspd, pspd))
    cases.append(_factory_case("spd_retr_second", "spd_retr", xspd, eta_p, yspd))
    cases.append(
        _factory_case(
            "spd_transp_id",
            "spd_transp",
            xspd,
            eta_p,
            eta_p,
            x_to=_floats(yspd),
        )
    )
    return cases


def _manopt_tree_runnable(root: Path) -> bool:
    """True when the tree has importmanopt or tools needed to call factories."""
    clone = _manopt_clone_root(root)
    markers = (
        clone / "importmanopt.m",
        clone / "manopt" / "importmanopt.m",
        clone / "manopt" / "tools" / "multisym.m",
        clone / "tools" / "multisym.m",
    )
    return any(path.is_file() for path in markers)


def _matlab_or_octave() -> tuple[str, list[str]] | None:
    octave = shutil.which("octave-cli") or shutil.which("octave")
    if octave:
        return "octave", [octave, "--quiet", "--no-window-system", "--eval"]
    matlab = shutil.which("matlab")
    if matlab:
        return "matlab", [matlab, "-batch"]
    return None


def _run_manopt_engine(root: Path, argv: list[str]) -> list[dict] | None:
    """Call factory proj/retr/transp from a manopt tree via MATLAB/Octave."""
    clone = _manopt_clone_root(root)
    tmp = Path(tempfile.mkdtemp(prefix="rgmin-manopt-gold-"))
    out = tmp / "manopt_gold.txt"
    root_lit = str(clone).replace("'", "''")
    out_lit = str(out).replace("'", "''")
    script = f"""
addpath(genpath('{root_lit}'));
if exist('importmanopt', 'file') == 2
    importmanopt;
end
fid = fopen('{out_lit}', 'w');
Ms = spherefactory(3);
xs = [0; 1; 0];
vs = [0.2; 0.3; -0.1];
ps = Ms.proj(xs, vs);
rs = Ms.retr(xs, ps);
ts = Ms.transp(xs, rs, ps);
fprintf(fid, '%.17g %.17g %.17g\\n', ps(1), ps(2), ps(3));
fprintf(fid, '%.17g %.17g %.17g\\n', rs(1), rs(2), rs(3));
fprintf(fid, '%.17g %.17g %.17g\\n', ts(1), ts(2), ts(3));
Mp = positivefactory(3);
xp = [1.5; 0.5; 2.0];
vp = [0.1; -0.2; 0.0];
pp = Mp.proj(xp, vp);
yp = Mp.retr(xp, vp);
tp = Mp.transp(xp, yp, vp);
fprintf(fid, '%.17g %.17g %.17g\\n', pp(1), pp(2), pp(3));
fprintf(fid, '%.17g %.17g %.17g\\n', yp(1), yp(2), yp(3));
fprintf(fid, '%.17g %.17g %.17g\\n', tp(1), tp(2), tp(3));
Msym = symmetricfactory(2);
xsym = [1, 0; 0, -1];
vsym = [0, 0.2; -0.1, 0];
psym = Msym.proj(xsym, vsym);
etas = [0, 0.05; 0.05, 0];
rsym = Msym.retr(xsym, etas);
tsym = Msym.transp(xsym, rsym, etas);
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', psym(1,1), psym(1,2), psym(2,1), psym(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', rsym(1,1), rsym(1,2), rsym(2,1), rsym(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', tsym(1,1), tsym(1,2), tsym(2,1), tsym(2,2));
Mc = complexcirclefactory(2);
zc = [1; 1i];
uc = [0.5+0.25i; -0.1+0.8i];
pc = Mc.proj(zc, uc);
rc = Mc.retr(zc, pc);
tc = Mc.transp(zc, rc, pc);
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', real(pc(1)), imag(pc(1)), real(pc(2)), imag(pc(2)));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', real(rc(1)), imag(rc(1)), real(rc(2)), imag(rc(2)));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', real(tc(1)), imag(tc(1)), real(tc(2)), imag(tc(2)));
Mm = multinomialfactory(3);
xm = [0.2; 0.3; 0.5];
vm = [1; 2; 3];
pm = Mm.proj(xm, vm);
etam = [0.1; -0.05; -0.05];
rm = Mm.retr(xm, etam);
tm = Mm.transp(xm, rm, etam);
fprintf(fid, '%.17g %.17g %.17g\\n', pm(1), pm(2), pm(3));
fprintf(fid, '%.17g %.17g %.17g\\n', rm(1), rm(2), rm(3));
fprintf(fid, '%.17g %.17g %.17g\\n', tm(1), tm(2), tm(3));
Mcm = centeredmatrixfactory(2, 2);
xcm = [1, -1; 0, 0];
vcm = [1, 3; 2, 4];
pcm = Mcm.proj(xcm, vcm);
etac = Mcm.proj(xcm, [0.2, -0.2; -0.1, 0.1]);
rcm = Mcm.retr(xcm, etac);
tcm = Mcm.transp(xcm, rcm, etac);
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', pcm(1,1), pcm(1,2), pcm(2,1), pcm(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', rcm(1,1), rcm(1,2), rcm(2,1), rcm(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', tcm(1,1), tcm(1,2), tcm(2,1), tcm(2,2));
Mspd = sympositivedefinitefactory(2);
xspd = eye(2);
vspd = [0.3, 1; -0.4, 0.5];
pspd = Mspd.proj(xspd, vspd);
etap = [0, 0.2; 0.2, 0];
rspd = Mspd.retr(xspd, etap);
tspd = Mspd.transp(xspd, rspd, etap);
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', pspd(1,1), pspd(1,2), pspd(2,1), pspd(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', rspd(1,1), rspd(1,2), rspd(2,1), rspd(2,2));
fprintf(fid, '%.17g %.17g %.17g %.17g\\n', tspd(1,1), tspd(1,2), tspd(2,1), tspd(2,2));
fclose(fid);
"""
    try:
        proc = subprocess.run(
            argv + [script],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not out.is_file():
        return None
    lines = out.read_text().strip().splitlines()
    if len(lines) != 21:
        return None
    try:
        nums = [[float(v) for v in line.split()] for line in lines]
    except ValueError:
        return None
    want = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    if [len(row) for row in nums] != want:
        return None
    published = {c["name"]: c for c in _published_factory_cases()}

    def _fill(name: str, s: list[float], v: list[float] | None = None) -> dict:
        case = dict(published[name])
        case["s"] = s
        if v is not None:
            case["v"] = v
        return case

    ps, rs, ts = nums[0], nums[1], nums[2]
    pp, yp, tp = nums[3], nums[4], nums[5]
    psym, rsym, tsym = nums[6], nums[7], nums[8]
    pc, rc, tc = nums[9], nums[10], nums[11]
    pm, rm, tm = nums[12], nums[13], nums[14]
    pcm, rcm, tcm = nums[15], nums[16], nums[17]
    pspd, rspd, tspd = nums[18], nums[19], nums[20]
    return [
        _fill("sphere_proj_north", ps),
        _fill("sphere_retr_north", rs, ps),
        _fill("sphere_transp_north", ts, ps),
        _fill("positive_proj_id", pp),
        _fill("positive_retr_exp", yp),
        _fill("positive_transp_id", tp),
        _fill("symmetric_multisym", psym),
        _fill("symmetric_retr_add", rsym),
        _fill("symmetric_transp_id", tsym),
        _fill("complexcircle_proj_pair", pc),
        _fill("complexcircle_retr_sign", rc, pc),
        _fill("complexcircle_transp_arrive", tc, pc),
        _fill("multinomial_proj_fisher", pm),
        _fill("multinomial_retr_exp", rm),
        _fill("multinomial_transp_arrive", tm),
        _fill("centered_proj_cols", pcm),
        _fill("centered_retr_add", rcm),
        _fill("centered_transp_id", tcm),
        _fill("spd_proj_symm", pspd),
        _fill("spd_retr_second", rspd),
        _fill("spd_transp_id", tspd),
    ]


def mint_manopt_formulas() -> tuple[dict, list[dict]]:
    """Mint factory numbers from MANOPT_ROOT when set, else published formulas.

    When MANOPT_ROOT points at a NicolasBoumal/manopt tree, the remint
    records the factory files it used. MATLAB/Octave, if present, runs
    those factories; otherwise the published Python formulas fill in.
    dest tests load only the frozen JSON and do not require this tree.
    """
    root = _manopt_root()
    if root is None:
        return {"manopt": "published-factory-formulas"}, _published_factory_cases()

    factories = _locate_manopt_factories(root)
    rels = [_rel_manopt_file(root, path) for path in factories]
    meta: dict = {
        "manopt": "MANOPT_ROOT",
        "manopt_files": rels,
        "manopt_exists": all(path.is_file() for path in factories),
    }
    engine = _matlab_or_octave() if _manopt_tree_runnable(root) else None
    if engine is not None:
        name, argv = engine
        cases = _run_manopt_engine(root, argv)
        if cases is not None:
            meta["manopt_engine"] = name
            return meta, cases
    meta["manopt_engine"] = "published-factory-formulas"
    return meta, _published_factory_cases()


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
