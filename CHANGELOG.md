# Changelog

All notable changes to CBTV are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-11-11

### Added
- `--piv-ambiguity-disc` (default 3 px): the half-width of the square around the PIV correlation peak that is
  excluded when the second peak is sought. Before, the 3 px was fixed in the code. It is in pixels, so it should be
  scaled with the image scale, like `--r-min`, `--r-max` and `--max-dist`. With the default, tracks are
  byte-identical to 1.0.0.

## [1.0.0] - unreleased

First public release. Until this release the code was developed in a private repository under the name
CPTV (Colour Particle Tracking Velocimetry); see `PROVENANCE.md`.

### Renamed
- The project is now **CBTV** (Colour Bean Tracking Velocimetry), formerly CPTV. The package is `cbtv`
  (was `cptv`), the command is `cbtv` (was `cptv`), the entry script is `run_cbtv.py` (was `run_cptv.py`) and
  the post-run cleaning script is `cbtv_clean.py` (was `cptv_clean.py`). Results are unchanged: the regression
  fixtures give byte-identical `tracks.csv` before and after the rename.
- The old `cptv` command still works and prints a one-line deprecation note. The debugging environment
  variables are now `CBTV_DUMP_C`, `CBTV_GS_DEBUG` and `CBTV_ASSIGN_PARITY`; the former `CPTV_*` names are
  still read.

### Changed (default)
- **The default tracking preset is now `bt`**, the published configuration plus `--coast-predict flow
  --exit-kill 1 --color-source track`. `--preset ship-v4` gives the configuration of the paper's first
  version (the same tracker arguments as the previous default). On the 71 benchmark cases, `bt` scores a
  mean HOTA of 0.7701 against 0.7367 for `ship-v4`. In the runs that selected it, no case was worse by more
  than 0.002. This was a selection made on the benchmark itself. `run_cbtv.SHIP_CONFIG` now aliases the default preset; use
  `SHIP_CONFIG_V4` for the published flags.

### Changed
- The detection code that the shipped detectors use now lives in `cbtv/detect/`. The research-only
  detection modules and the `sys.path` inserts in the detector adapters are gone, and `matplotlib` is no
  longer a dependency. Detections and tracks are byte-identical.
- `--gpu 0` now runs on CPU only: OpenCV's OpenCL path is switched off, so no environment variables are
  needed for CPU runs.
- The `cbtv` command runs the tracker in your current directory, so relative paths work, and writes the
  tracker's debug folder into the output directory.
- Optional components that are missing now print a one-line `[WARN]` (no CuPy, `--gpu auto` falling back to
  CPU, no `tqdm`). Required ones fail with a clear error instead of silently changing results.

### Added
- `pip install .` packaging (`pyproject.toml`) with the `cbtv` command and the optional extras `[gpu]` (CuPy
  for CUDA 12), `[cellpose]`, `[test]` and `[api]`.
- `cbtv video.mp4`: the video can be given without `--input`. `--preset` and `--version`.
- `--seed` (default 0): every random draw on the tracking and detection path comes from it, so the same seed
  and inputs give the same `tracks.csv`.
- Input errors are reported before tracking starts (exit code 2, one line; full traceback with
  `--verbose 1`): a missing or unreadable video, `--gpu 1` without a working CuPy, a missing, unreadable or
  malformed detections CSV, no detections in the first 5 frames, a median detection radius far outside
  `--r-min/--r-max` (with a suggested range), and `--detections-csv` combined with a non-`csv` detector.
- `--detections-csv` on its own now selects the `csv` detector.
- `CBTV_DUMP_C=<dir>` environment variable: writes each frame's final cost matrix to `<dir>/C_<frame>.npz`
  for debugging; no effect when unset.
- The Python API `cbtv.track()` (needs the `[api]` extra for its pandas output).
- Regression fixtures with reference outputs (`tests/fixtures/`, `tools/run_fixtures.py`).
- Two example videos from the authors' simulations, with detections and READMEs: `examples/impulsive_collision`
  and `examples/inclined_chute`.
- In/outflow handling for particles that enter and leave the view (`--coast-predict`, `--exit-kill`,
  `--region-map`, `--occluder-coast`, `--confirm-hits`, `--row-tags`), the post-run cleaning script
  `cbtv_clean.py` and the QA overlay `render_states.py`.

### Fixed
- **`--exit-kill` (default `bt` preset) ended tracks of particles resting at the frame edge**, including
  particles detected just past it. On real footage with particles against the frame border about 0.7% of
  links were lost, and tracks born there did not survive. Exit-kill now ends only tracks that are leaving
  (coasting, or last matched inside the view) and keeps detections within `--exit-margin` of the edge; exit
  regions of a `--region-map` are unchanged. On the 71 benchmark cases the results differ from the earlier
  numbers by at most 0.0021 PTR, and 64 cases are identical.
- **CPU and GPU runs could choose different radius windows** at frame 0 (the automatic `--r-min/--r-max`
  check): the radii were estimated with the frame-0 Laplacian-of-Gaussian detector, whose CPU and GPU
  implementations differ. With `--detector csv` the radii now come from the frame-0 rows of the detections
  CSV (rescaled into the rule's units, `CSV_TO_LOG_RADIUS` = 4.5 / 6.555); other detectors use the CPU
  implementation on every device. CPU and GPU now give byte-identical `tracks.csv` on all 71 benchmark
  cases (`bt`). The paper's numbers were regenerated with this fix and the exit-kill fix above.
- **Edge-fit sub-pixel refinement was silently disabled** in the first standalone copy (a research-only
  module was missing), which changed every result. It is now part of `cbtv/` and always on, as in the
  published runs.
- **CPU runs did not match GPU runs**: the CPU cost matrix scored fewer track-detection pairs than the GPU
  one. CPU and GPU now give byte-identical `tracks.csv` on the regression fixtures. Sorts on the tracking
  path are now stable, so the result no longer depends on the CPU's SIMD instruction set.
- The `bayes_adapt` detector gave different results on every run (an unseeded random shuffle).
- The default detector (`ws_multi_vote`) returned zero detections on every frame when its GPU strategy
  module failed to import; it now stops with an error naming the missing file.
- A missing `scipy` silently skipped the final (Hungarian) assignment stage; `scipy` is now a hard
  requirement.
- An argparse help string crashed the tracker at start-up on Python 3.14.
- **`--second-pass 1` crashed** with `TypeError` on the first frame that reached the second matching pass (the call
  to the cost matrix omitted one argument). The option is off by default and in both presets.
- **A CUDA installation that could not compile kernels** (for example a missing `nvrtc-builtins` library) passed the
  GPU check, because a cached kernel hid the problem, and the run then crashed partway through. The check now
  compiles a kernel, so `--gpu auto` and the tracker core fall back to the CPU, and `--gpu 1` stops with a clear
  error. The fallback gives the same tracks as a CPU run.
- `--gpu 1` with CuPy installed but no CUDA device crashed after announcing the CPU fallback; it now runs on the CPU.
- Ctrl+C was ignored inside the kernel-recovery step (two bare `except:` clauses).
