- Added an opt-in FIRE 2.0 stabilization that caps `dt_max` when an uphill
  reset follows one downhill move, preventing repeated semi-implicit-Euler
  overshoot cycles on stiff systems. The Guénolé 2020 schedule leaves it
  off by default.
- Added per-step FIRE diagnostic tracing (`FireTrace`,
  `Solver::start_fire_trace`, `Solver::take_fire_trace`) recording `dt`,
  `dt_max`, `alpha`, `power`, `n_pos`, `half_step_backs`, and
  `clipped_moves`.
- Added C ABI and Python toggles for `cap_two_step_overshoot`
  (`rgmin_solver_set_fire_cap_two_step_overshoot`, `xts` alias, C++
  wrapper, and `Session(..., cap_two_step_overshoot=...)`), and bumped
  `RGMIN_ABI_VERSION_MINOR` from 47 to 48.
