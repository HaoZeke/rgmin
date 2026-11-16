The contributing guide asks for a towncrier fragment instead of an edit
to `CHANGELOG.md`, and names the fragment types.
It lists the clippy commands that CI runs with warnings denied on Rust
1.88.0 and stable.

The Unreleased sections stay empty, and `scripts/check_changelog.py` fails when a fragment is paired with an Unreleased line.
