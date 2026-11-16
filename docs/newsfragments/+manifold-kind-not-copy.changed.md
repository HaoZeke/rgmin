`ManifoldKind` is no longer `Copy`, since a product and a subspace sphere share their geometry through an `Arc`; clone it where a copy was implied.
`stiefel_p` and `as_str` take `&self`, and the new `Product` and `SubspaceSphere` variants extend exhaustive matches on the kind.
