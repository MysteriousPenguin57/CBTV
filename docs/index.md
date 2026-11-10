# CBTV documentation

CBTV (Colour Bean Tracking Velocimetry) tracks multicoloured particles in videos of dense granular flows. It writes
per-particle trajectories (`tracks.csv`), an annotated video and a short summary of the run.

| page | contents |
|---|---|
| [Installation](install.md) | pip, conda, extras, the pinned environment of the paper, and the GPU runtime |
| [User guide](user_guide.md) | options, presets, detectors, tuning, particles entering and leaving the view, the `tracks.csv` format, CPU and GPU, troubleshooting |
| [Five-minute tutorial](tutorial_fixture.md) | run CBTV on the bundled 60-frame fixture and check the output against the reference |
| [Python API](api_python.md) | `cbtv.track()` |
| [Module map](api.md) | what each file in `cbtv/` does, and the entry points |

See also the [README](../README.md), the [CHANGELOG](../CHANGELOG.md), [CONTRIBUTING](../CONTRIBUTING.md) and
[CITATION.cff](../CITATION.cff).

Licence: BSD-3-Clause ([LICENSE](../LICENSE)).
