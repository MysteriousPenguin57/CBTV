# CBTV: Colour Bean Tracking Velocimetry

CBTV tracks multicoloured particles in videos of dense granular flows. It first matches constellations of
neighbouring coloured particles as single units, guided by a motion prior that is measured by cross-correlation for
slow particles and extrapolated from each track's history for fast ones, and then assigns the remaining particles
individually. The accompanying paper compares CBTV with ten tracking methods on discrete-element simulations of
granular flows. Its advantage, measured as the share of particle identities kept from frame to frame, lies in dense
flows in which particles move between about half and twice their spacing per frame, where links between frames are
ambiguous. The method needs no training data.

For each video, CBTV writes the trajectory of every particle (`tracks.csv`), an annotated video with each particle
circled and labelled with its track ID, and a short summary of the run. Detection is separate from linking particle
identities from frame to frame, so you can also give CBTV your own detections.

## System requirements

- Python 3.10 or newer.
- Tested on Windows 11 (CPU and NVIDIA GPU) and on Linux (Ubuntu, CPU). On macOS with Apple silicon, CBTV runs, but
  the tracks differ numerically from those on Windows and Linux (see "Platforms" in the
  [user guide](docs/user_guide.md#platforms)).
- A GPU is optional. CPU and GPU runs give the same tracks; the GPU is faster. GPU acceleration needs an NVIDIA GPU
  with CUDA 12 and CuPy (the `[gpu]` extra below).

## Installation

From GitHub, with pip:

```
pip install "git+https://github.com/MysteriousPenguin57/CBTV.git"
```

From a clone of the repository:

```
git clone https://github.com/MysteriousPenguin57/CBTV.git
cd CBTV
pip install .               # or, for development: pip install -e ".[test]"
```

Both install the `cbtv` command. `cbtv --version` checks the install. The optional extras are:

| extra | adds | needed for |
|---|---|---|
| `[gpu]` | `cupy-cuda12x` | GPU acceleration (CUDA 12) |
| `[cellpose]` | `cellpose` | the `cellpose` detector |
| `[test]` | `pytest`, `pandas`, `hypothesis` | the test suite |
| `[api]` | `pandas` | the Python function `cbtv.track()` |

For example, `pip install ".[gpu]"` from a clone.

With conda or mamba, from a clone:

```
conda env create -f environment.yml          # or: mamba env create -f environment.yml
conda activate cbtv
pip install --no-deps .
```

`requirements.txt` lists the core dependencies, for running `python run_cbtv.py` from the repository folder without
installing. `requirements-lock.txt` pins the exact versions used for the paper (see "Reproducing the paper's
configuration" below). [docs/install.md](docs/install.md) covers these options in more detail, including the CUDA runtime that CuPy needs.

## Quick start

The repository includes two short example videos from the authors' discrete-element simulations, each with its
particle detections and a README:

| example | flow | frames | run time (CPU) |
|---|---|---|---|
| [`examples/impulsive_collision`](examples/impulsive_collision/README.md) | two dense blobs of 534 particles colliding | 101 | 2 to 3.5 min |
| [`examples/inclined_chute`](examples/inclined_chute/README.md) | dense flow down a 25 degree chute | 120 | 2.5 to 3.5 min |

The examples are not part of the pip package, so clone or download the repository first. Then, from the repository
folder:

```
cd examples/impulsive_collision
cbtv clip.mkv --detections-csv detections.csv --gpu 0
```

`--detections-csv` gives CBTV the example's detections, and `--gpu 0` runs on the CPU (use `--gpu 1` for a CUDA GPU;
the tracks are the same). The run takes a few minutes on a CPU and writes into `results/`:

| file | contents |
|---|---|
| `tracks.csv` | one row per particle per frame: `frame, track_id, x, y, r, missed` |
| `annotated.mp4` | every particle circled in a persistent per-track colour, with its track ID |
| `overlay.mp4` | the tracker's diagnostic overlay (detections, live tracks, coasting tracks) |
| `summary.txt` | track counts and lengths |
| `cluster_log.csv` | per-frame statistics of the constellation stage |

`summary.txt` should start with:

```
frames processed          : 101
unique tracks             : 538
track length median/mean  : 101 / 100.3 frames
tracks spanning full video: 530 (98.5%)
particles per frame median: 534
coasting rows (missed>0)  : 38 (0.1% of rows)
```

530 of the 534 particles keep one identity through all 101 frames. Each example's README says what its result
should look like.

## Your own video

```
cbtv my_experiment.mp4 --r-min 6 --r-max 10
```

This detects particles with the default detector (`ws_multi_vote`) and tracks them with the default preset (`bt`).
Set `--r-min` and `--r-max` to bracket the particle radii in your video, in pixels; a wrong radius range is the most
common cause of poor results. Check the settings on the first 50 frames first (`--max-frames 50`), and look at
`overlay.mp4` to see whether the particles are detected.

If you have your own detections, give them as a CSV file with one row per detected particle:

```
frame,x,y,r,score
0,700.865,158.986,6.699,0.9824
0,402.713,440.652,6.050,0.9899
```

`frame` is the processed frame index (0, 1, 2, ... after any `--frame-step`), `x` and `y` are the centre in pixels
(origin at the top left), `r` is the radius in pixels, and `score` (optional) is the detector's confidence. Then run
`cbtv my_experiment.mp4 --detections-csv my_detections.csv`. CBTV still reads the video, because it uses the
particle colours. If your detections are already checked, add `--extra "--csv-trust 1"`, so that CBTV keeps all of
them instead of filtering them by image contrast.

The [user guide](docs/user_guide.md) covers the detectors, tuning, bright backgrounds, particles entering and leaving
the view, and how to read `summary.txt`.

## Command line

| option | default | meaning |
|---|---|---|
| (video) | | the input video |
| `--preset` | `bt` | tracking configuration (`bt` or `ship-v4`) |
| `--detector` | `ws_multi_vote` | detection method; `csv` with `--detections-csv` |
| `--detections-csv` | | your own detections (`frame,x,y,r[,score]`) |
| `--r-min`, `--r-max` | 6, 10 | particle radius range in pixels |
| `--gpu` | `auto` | `auto`, `1` (GPU) or `0` (CPU) |
| `--frame-step`, `--max-frames` | 1, -1 | process every Nth frame; stop after N frames |
| `--seed` | 0 | random seed |
| `--output-dir` | `results` | output folder |
| `--extra` | | further tracker options, as one quoted string |

`cbtv --help` lists all options of the command, and `python -m cbtv.run_tracker --help` those of the tracker core.
The [user guide](docs/user_guide.md) describes the options, presets and detectors, tuning for your own video,
particles entering and leaving the view, the output format, and troubleshooting. The Python function `cbtv.track()` is
described in [docs/api_python.md](docs/api_python.md).

## Reproducing the paper's configuration

All CBTV results in the paper use the default preset, `bt`. The configuration as first frozen for the paper is
available as `--preset ship-v4`. Every random draw comes from `--seed` (default 0), so the same inputs and seed give
the same `tracks.csv`.

To reproduce the reference outputs in `tests/fixtures/` byte for byte, install the pinned environment into a new,
empty environment:

```
python -m venv .venv
.venv\Scripts\activate                        # Linux/macOS: source .venv/bin/activate
set PYTHONNOUSERSITE=1                        # Linux/macOS: export PYTHONNOUSERSITE=1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

With this environment, CPU and GPU runs gave byte-identical tracks on all 71 benchmark cases of the paper on Windows,
and Linux (CPU) matched Windows on the 10 cases checked. The benchmark data and the scripts behind the paper's tables
and figures are not part of this repository.

## Running the tests

```
pip install -e ".[test]"
pytest -m "not gpu"            # CPU tests
pytest -m gpu                  # GPU tests; need CuPy and a CUDA GPU
```

The regression tests run the tracker on three short fixtures in `tests/fixtures/` and compare the result byte for
byte with the reference outputs there. They take a few minutes on a CPU; `pytest -m "not gpu and not slow"` skips
them. If pytest
cannot write to its default temporary folder on Windows, add `--basetemp <writable folder>`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for more.

## How to cite

If you use CBTV, please cite the paper:

> K. Thurairajah and Z. Pan, "CBTV: Colour Bean Tracking Velocimetry for Dense Granular Flows", arXiv:XXXX.XXXXX.

and the version of the software you used:

> K. Thurairajah, CBTV (version 1.0.0), Zenodo, doi:10.5281/zenodo.XXXXXXX.

The same information is in [CITATION.cff](CITATION.cff), from which GitHub shows a "Cite this repository" entry.

## Licence

CBTV, including the example videos and detections, is released under the BSD 3-Clause licence; see [LICENSE](LICENSE).

## Contact

Kaveeshan Thurairajah (kavee@thurairajah.org), Department of Mechanical and Mechatronics Engineering, University of
Waterloo. Please report bugs and questions as GitHub issues.
