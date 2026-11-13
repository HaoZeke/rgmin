FIRE
====

``Method::Fire { kind: FireKind::V2 }`` is FIRE 2.0.
``Solver::set_fire_preset(FirePreset::Guenole2020)`` installs the
published parameter set. ``FireConfig::ase`` is the ASE constructor
(``dtmax = 1``) rather than the session default ``dt_max = 2.5 dt``.

The program is ``examples/tutorial_fire.rs``. Continuous integration runs it.

The Python class ``rgmin.FIRE`` passes the ASE arguments (``dt``,
``dtmax``, ``Nmin``, ``finc``, ``fdec``, ``astart``, ``fa``) into
``FireConfig``.
