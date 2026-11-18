The C header, `src/ffi.rs`, the manifold modules and the manifold how-tos no longer call `rgmin_manifold_t` tokens 7-10 reserved.
They are SPD, Grassmann, hyperboloid and Poincare ball, and token 23 is U(n). `rgmin_solver_set_manifold` has always mapped them.
