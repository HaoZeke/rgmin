`python -m pip install rgmin` installs the nanobind package from PyPI.
A `vX.Y.Z` tag builds the sdist and the wheels for Linux (`manylinux_2_28`) on x86_64 and aarch64, macOS 14 or newer on arm64 and x86_64, and Windows on x86_64, tests every wheel, and uploads them to PyPI through trusted publishing.
CPython 3.10 and 3.11 each get a wheel, one abi3 wheel serves 3.12 and newer, and the free-threaded 3.14t and 3.15t each get a wheel.
The package version comes from `Cargo.toml`, and the upload stops unless the tag names that version.
