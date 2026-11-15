`rgmin.optimize.Adam` takes a `learning_rate` (default 0.02 Å), the step each coordinate makes per step.
Adam divides out the gradient's scale, so `alpha` never sized its step: every step was the 0.2 Å `maxstep` cap.
At the cap Adam climbed out of the start basin and walked off a `StrainFilter`'s strain, and where a cell relaxation ended depended on the last bits of the forces, so `UnitCellFilter` runs reached another minimum on some machines.
`rgmin._core.Session` takes `istep`, the step scale of the methods other than FIRE.
