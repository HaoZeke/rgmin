The Python tests pass on macOS x86_64, where Apple's libm `exp` rounds one value on the Pt7 EMT trajectory down and glibc rounds it up.
The Pt7 EMT digest, like the LJ38 one, is pinned on glibc only, and every other platform compares the Pt7 frames and energies with `tests/python/data/pt7_golden.json` within 1e-12, absolute in positions and relative in energies.
Every platform still requires the Pt7 runs at 1, 2 and 8 Python threads and at 1, 2, 3 and 8 Rayon threads to match a reference run on the same machine bit for bit.
Windows takes the same comparison, so `PT7_EMT_WINDOWS` and the dispatch-only `windows-pt7-digest` CI job that recorded it are gone.
