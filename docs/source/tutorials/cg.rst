Conjugate gradients
====================

Two steps share the Polak-Ribière coefficient. ``Method::eon_cg`` is
that coefficient with the Powell restart at 0.5, and the session still
line-searches. ``minimize_eon_cg`` is the client step: one curvature
probe, no line search, then the per-atom cap.

The program is ``examples/tutorial_cg.rs``. Continuous integration runs it.

``Method::polak_ribiere_plus`` is SciPy's ``method="CG"``: ``max(0, β)``.
The Python session named ``cg`` uses that coefficient with a strong Wolfe
search, because SciPy does. The eOn client step does not.
