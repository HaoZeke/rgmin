# MIT License
# Copyright 2026 Rohit Goswami
"""Drop-in replacements for ``ase.optimize.BFGS``, ``LBFGS`` and ``FIRE``.

The numerical step is an rgmin ``Solver``. ``LBFGS`` without
``use_line_search`` shortens the quasi-Newton step to ``maxstep`` and
keeps the pairs, which is what ASE does. ``eon_cap=True`` selects the
crate default instead: the line-search-free cap that clears a pair
when the proposal is longer than ``maxstep``. On the 10-dimensional
Rosenbrock start that well at ``x0 ≈ -1`` traps the ASE rule and not
the cap. ``alpha`` is accepted so existing call sites keep working;
the dense BFGS session starts from the identity, not ``alpha * I``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rgmin._core import Session


def _optimizer_base():
    from ase.optimize.optimize import Optimizer

    return Optimizer


class _RgminOptimizer(_optimizer_base()):
    """Shared ASE loop. Subclasses choose the rgmin method."""

    kind = "lbfgs"

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )
        self._native: Session | None = None

    def _open(self) -> Session:
        if self._native is None:
            count = len(self.atoms) * 3
            self._native = self._session(count)
        return self._native

    def _session(self, n: int) -> Session:
        return Session(self.kind, n)

    def _objective(self, x: list[float]) -> tuple[float, list[float]]:
        coords = np.asarray(x, dtype=np.float64).reshape((-1, 3))
        self.atoms.set_positions(coords, apply_constraint=False)
        energy = float(self.atoms.get_potential_energy())
        forces = np.asarray(self.atoms.get_forces(), dtype=np.float64)
        gradient = (-forces).reshape(-1)
        return energy, gradient.tolist()

    def step(self, forces: Any = None) -> None:  # noqa: ARG002
        native = self._open()
        x0 = np.asarray(self.atoms.get_positions(), dtype=np.float64).reshape(-1)
        updated, _energy, _gnorm = native.step(x0.tolist(), self._objective)
        self.atoms.set_positions(
            np.asarray(updated, dtype=np.float64).reshape((-1, 3)),
            apply_constraint=True,
        )


class BFGS(_RgminOptimizer):
    """Dense BFGS with the ASE ``BFGS`` constructor.

    ``alpha`` is the ASE initial-Hessian guess. This session starts
    from the identity. Both reach the same EMT minimum; the first
    step is not ASE's.
    """

    kind = "bfgs"

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        alpha: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.maxstep = 0.2 if maxstep is None else float(maxstep)
        self.alpha = 70.0 if alpha is None else float(alpha)
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )

    def _session(self, n: int) -> Session:
        return Session("bfgs", n, max_atom=self.maxstep)


class LBFGS(_RgminOptimizer):
    """Limited-memory BFGS with the ASE ``LBFGS`` constructor.

    ``use_line_search=False`` (the ASE default) is
    :class:`rgmin.LbfgsStep.Scaled <rgmin>`: shorten to ``maxstep``,
    keep the pairs. ``use_line_search=True`` is the strong Wolfe
    search. ``eon_cap=True`` is the Rust default and wins over the
    line search.
    """

    kind = "lbfgs"

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        maxstep: float | None = None,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        use_line_search: bool = False,
        eon_cap: bool = False,
        **kwargs: Any,
    ) -> None:
        self.maxstep = 0.2 if maxstep is None else float(maxstep)
        self.memory = int(memory)
        self.damping = float(damping)
        self.alpha = float(alpha)
        self.use_line_search = bool(use_line_search)
        self.eon_cap = bool(eon_cap)
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            **kwargs,
        )

    def _session(self, n: int) -> Session:
        cap = self.maxstep * self.damping if self.damping > 0.0 else self.maxstep
        return Session(
            "lbfgs",
            n,
            max_atom=cap,
            memory=self.memory,
            wolfe=self.use_line_search and not self.eon_cap,
            eon_cap=self.eon_cap,
        )


class FIRE(_RgminOptimizer):
    """FIRE with the ASE ``FIRE`` constructor.

    ``dtmax`` is applied. A fresh Rust session would cap the step at
    ``2.5 * dt``; :meth:`rgmin.FireConfig.ase` uses ``1.0``, and this
    constructor passes the caller's value. ``position_reset_callback``
    is accepted and not invoked: the velocity reset stays in the session.
    """

    kind = "fire"

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        dt: float = 0.1,
        maxstep: float | None = None,
        maxmove: float | None = None,
        dtmax: float = 1.0,
        Nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.1,
        fa: float = 0.99,
        a: float = 0.1,
        downhill_check: bool = False,
        position_reset_callback: Any = None,
        **kwargs: Any,
    ) -> None:
        step = maxstep if maxstep is not None else maxmove
        self.maxstep = 0.2 if step is None else float(step)
        self.dt = float(dt)
        self.dtmax = float(dtmax)
        self.Nmin = int(Nmin)
        self.finc = float(finc)
        self.fdec = float(fdec)
        self.astart = float(astart)
        self.fa = float(fa)
        self.a = float(a)
        self.downhill_check = bool(downhill_check)
        self.position_reset_callback = position_reset_callback
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            **kwargs,
        )

    def _session(self, n: int) -> Session:
        return Session(
            "fire",
            n,
            max_atom=self.maxstep,
            dt=self.dt,
            dt_max=self.dtmax,
            n_min=self.Nmin,
            f_inc=self.finc,
            f_dec=self.fdec,
            alpha=self.astart,
            f_alpha=self.fa,
            downhill=self.downhill_check,
        )
