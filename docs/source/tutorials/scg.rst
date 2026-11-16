

Scaled conjugate gradient
-------------------------

Møller SCG prices the step with a damped model. There is no line search. A positive finite ``Control::maxmove`` bounds every trial, including the retry after a non-finite value. Zero, negative, infinite and NaN leave the step unbounded.

The program is ``examples/tutorial_scg.rs``. Continuous integration runs it.

.. code:: rust

    use rgmin::{Conjugacy, Restart, ScgParams};

    let params = ScgParams::default();
    let _ = (params.sigma0, Conjugacy::PolakRibiere, Restart::Never);

The example calls ``minimize_scg`` on a quadratic and checks both coordinates.
