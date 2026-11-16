`scripts/check_changelog.py` fails when `docs/newsfragments/` holds a fragment and the Unreleased section of `CHANGELOG.md` or `docs/orgmode/changelog.org` also holds a line.
A file there whose type `towncrier.toml` does not name fails the same way.
`towncrier build --draft` renders the fragments and changes no file.
