# Provenance

CBTV was developed under the name CPTV (Colour Particle Tracking Velocimetry) in the authors' private research
repository, and was then moved into a private release repository on 2026-10-01. This public repository is a clean
copy of that release repository (commit `b8e365d`), renamed from CPTV to CBTV. Internal planning notes, ticket
reports, a draft software paper, a demo clip of experimental footage, and tools that need the authors' private
research data were left out. The rename changed no results: the regression fixtures in `tests/fixtures/` give
byte-identical `tracks.csv` before and after it.

The numbers in the paper were produced on GPU with the `ship-v4` and `bt` presets. The tracker code (`cbtv/`,
`run_cbtv.py`) is the code that produced them, apart from the rename and two fixes made for the public release
(see `CHANGELOG.md`): the `--second-pass` option no longer crashes, and a CUDA installation that cannot compile
kernels now falls back to the CPU. Neither changes the tracks of the two presets. Version 1.0.1 adds one option,
`--piv-ambiguity-disc`, which makes a previously fixed 3 px setting of the PIV peak test adjustable; at its default
of 3 it does not change the tracks.

The example videos in `examples/` are renderings of the authors' discrete-element simulations, two of the flows in
the paper, with the Cellpose detections that every tracker received in the paper's comparison.
