`Control` and `Report` each gain a public `stop_norm` field, and `StopNorm` gains the `Rms` variant.
Set it with `Control::with_stop_norm`; an exhaustive `match` on `StopNorm` needs an `Rms` arm.
