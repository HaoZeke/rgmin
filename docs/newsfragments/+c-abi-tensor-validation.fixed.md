`rgmin_minimize_hess_opts`, `rgmin_minimize_hvp_opts`,
`rgmin_minimize_eindir` and `rgmin_minimize_scg_with_options` return
`RGMIN_INVALID_PARAMETER` when the initial `x` is not finite, as
`rgmin_minimize` does. A block tensor import such as the seed of
`rgmin_lowest_eigenpairs` rejects a DLPack version other than 1.0 with the
same status.
