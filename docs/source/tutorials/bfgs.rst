BFGS
====

Dense BFGS keeps an inverse Hessian, starting from the identity.
Himmelblau has four minima. From ``(0, 0)`` the session reaches
``(3.584, -1.848)``, not ``(3, 2)``.

The program is ``examples/tutorial_bfgs.rs``. Continuous integration runs it.

ASE's ``BFGS`` multiplies the first step by ``1/alpha`` with ``alpha = 70``.
This session does not. The EMT gold in ``python/tests/test_ase_optimizer.py``
checks the minimum, not that first step.
