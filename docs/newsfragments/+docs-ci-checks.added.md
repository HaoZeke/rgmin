CI checks that every C function in `include/rgmin.h` has a doc comment (`scripts/check_header_docs.py`), that every DOI and URL in the tracked sources resolves (`scripts/check_links.py`), and that rustdoc builds with warnings as errors.
The docs workflow builds with `-W` and fails when the tracked rst under `docs/source` is stale.
