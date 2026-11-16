Python
======

The package is ``python/``. Install it with maturin. ASE is optional and
imported when an optimizer is constructed.
:doc:`ase-migration` is the one-line import change and the differences from ASE.

.. code:: bash

   pip install "python/[ase]"

.. code:: python

   from ase.build import molecule
   from ase.calculators.emt import EMT
   from rgmin import FIRE, LBFGS

   atoms = molecule("H2")
   atoms.calc = EMT()
   opt = LBFGS(atoms, trajectory="opt.traj", logfile=None)
   opt.attach(lambda: atoms.get_potential_energy())
   opt.run(fmax=0.05)

``LBFGS(eon_cap=True)`` selects the Rust default, the line-search-free
eOn cap. ``LBFGS(wolfe=True)`` selects the strong Wolfe search. Without
either the class matches ASE's length control. Stubs live in
``python/rgmin`` and ``py.typed`` marks the package.

In the nanobind wheel, ``SolverConfig("lbfgs")`` and ``rgmin.ase.LBFGS``
take the eOn step by default. Its first step spends one extra
calculator call on a finite-difference probe 0.01 along the force. The
probe sets the initial inverse curvature and does not move the atoms.
Later steps cost one call each. A step whose largest atom move exceeds
``maxstep`` is scaled down whole, as ASE does, so no atom moves farther
than ``maxstep``. A non-positive ``maxstep`` restores the eOn cap of
0.2. ``linesearch="wolfe"`` opts into the strong Wolfe search, with
``c1``, ``c2``, and ``linesearch_maxiter`` as its constants, and
``maxstep`` still bounds every atom move. The eOn step reads no other
``linesearch`` value, so ``brent``, ``backtracking``, and ``goldstein``
raise ``ValueError`` on ``lbfgs``.

.. code:: python

   from rgmin.ase import LBFGS

   LBFGS(atoms, logfile=None, maxstep=0.1).run(fmax=0.05)
   LBFGS(atoms, logfile=None, linesearch="wolfe").run(fmax=0.05)

``linesearch`` is accepted only where the step reads it. ``sd``, the
conjugate-gradient methods, ``bfgs``, ``adam``, and ``diis`` take every
kind, and ``lbfgs`` takes ``wolfe``. ``accept="step"`` takes each step
without a line search, and FIRE, QuickMin, ODE12r, BB, SR1, SR2, PSO,
Newton, RFO, dogleg, and Newton-CG never search a line. A
``linesearch`` on any of those raises ``ValueError`` and names the
reason, rather than leaving the run unchanged. ``rgmin.ase`` takes
``accept="step"`` on a non-conservative optimizable (NEB, dimer,
min-mode), so a ``linesearch`` there raises unless ``accept`` is
``energy`` or ``nonmonotone``.

``lowest_eigenpairs(hvp, x0_block)`` returns values, vectors, residual
norms, and a converged flag for each of the k lowest pairs.
``hvp(x, v)`` is ``H(x) v``. ``precond`` applies ``T`` to a block.
``strict=False`` returns the flags instead of raising.
:doc:`../tutorials/eigenpairs` is the short example, and continuous
integration runs it.

C and C++ use ``include/rgmin.h`` and ``include/rgmin/optimize.hpp``.
The C session is a handle: ``rgmin_solver_create``, ``rgmin_solver_step``,
``rgmin_solver_free``. ``rgmin_solver_dim`` and ``rgmin_solver_steps``
read that handle. ``rgmin_last_error_message`` is the same borrowed
string as ``rgmin_last_error``.
