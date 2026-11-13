From ASE, SciPy and eOn
=======================

The Rust default is the eOn L-BFGS cap (``EON_MAX_MOVE = 0.2``), not a
line search. ``LbfgsStep::Wolfe`` opts into strong Wolfe.
``LbfgsStep::Scaled`` is ASE ``LBFGS`` with ``use_line_search=False``:
shorten the step, keep the pairs.

.. list-table::
   :header-rows: 1

   * - Call site
     - rgmin
   * - ASE ``BFGS(atoms, trajectory=, logfile=)``
     - ``rgmin.BFGS``. The initial Hessian is the identity, not ``alpha * I``.
   * - ASE ``LBFGS(..., use_line_search=False)``
     - ``rgmin.LBFGS``. Same length control.
   * - ASE ``LBFGS(..., use_line_search=True)``
     - ``rgmin.LBFGS(use_line_search=True)``, the strong Wolfe search.
   * - ASE ``FIRE(dt=, dtmax=, Nmin=, ...)``
     - ``rgmin.FIRE``. ``dtmax`` is kept. A Rust session would use ``2.5 * dt``.
   * - eOn ``lbfgs``, ``max_move = 0.2``
     - ``Lbfgs::default`` or ``rgmin.LBFGS(eon_cap=True)``.
   * - eOn ``cg``
     - ``minimize_eon_cg``. ``Method::eon_cg`` is only the coefficient and the restart.
   * - SciPy ``method="L-BFGS-B"``
     - ``Lbfgs::minimize_box``. ``LBFGSB_FTOL`` is SciPy's ``ftol``.
   * - SciPy ``method="CG"``
     - ``Method::polak_ribiere_plus``, ``max(0, β)``.

``run(fmax, steps)``, ``attach``, ``trajectory`` and ``logfile`` are ASE's.
The classes subclass ``ase.optimize.optimize.Optimizer``.

The scaled step is not the one to copy when the function has a flat well.
On the 10-dimensional Rosenbrock start, ASE's length control stops near
``x0 = -1`` with value about 4 and an infinity norm below ``1e-5``. The
eOn cap and the Wolfe search both reach the minimum at 0. SciPy's
L-BFGS-B does too. ``tests/scaled_rosenbrock10.rs`` records that. The
tolerance is the same; the step rule is what leaves the well.

``Stepper::step`` is the one-iteration seam rgsaddle already calls on
``Solver``.
