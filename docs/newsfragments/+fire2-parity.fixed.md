- Non-finite starts and trials now fail before any clip bookkeeping is
  recorded in `step_first_order` (`src/session.rs:3710`) and `assess`
  (`src/session.rs:5237`), and `clip_free` (`src/session/constrained.rs:97`)
  keeps its finite-only coordinate clipping.
- ABC-FIRE component clipping now leaves the velocity unchanged when
  `vmax <= 0` in `fire_clip_components` (`src/fire.rs:609`) instead of
  forcing every component to zero.
