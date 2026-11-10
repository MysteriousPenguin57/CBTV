# Module map

CBTV runs from the command line (`cbtv`) or from Python ([`cbtv.track(...)`](api_python.md), which returns the
tracks as a pandas DataFrame and runs the same code path). This page maps the code so you can find your way around it.

## Entry points

| entry point | what it does |
|---|---|
| `cbtv` command = `run_cbtv.py:main` | the user-facing wrapper: checks inputs, runs the tracker with a preset (`PRESETS`: `bt` default = `BT_CONFIG`, `ship-v4` = `SHIP_CONFIG_V4`), renders `annotated.mp4`, writes `summary.txt` |
| `cbtv.track(...)`, `cbtv.presets()` = `cbtv/api.py` | the Python API: keyword arguments -> the `cbtv` command's parser, checks and `run`; returns `tracks.csv` as a DataFrame ([page](api_python.md)) |
| `python -m cbtv.run_tracker` = `cbtv/run_tracker.py:main` | the tracker core: every option (`--help`; definitions in `cbtv/config.py`); writes `tracks.csv` and `overlay.mp4` |
| `cbtv_clean.py`, `render_states.py` | post-run cleaning of `tracks.csv` and a QA overlay coloured by row state (user guide) |

## Pipeline, per frame

1. **Detect**: `detectors.run_detector` dispatches to a preset; the detections then pass the multi-stage filter.
2. **Predict motion**: the PIV-style dense motion prior (`piv.py`) or each track's own velocity.
3. **Constellations**: colour-coherent clusters matched as units commit their members (`clusters.py`).
4. **Cost matrix + assignment** for the rest (`cost_matrix.py`, `match_ops.py`, `assignment.py`).
5. **Track lifecycle and output**: update, coast, end, write rows (`tracks.py`, `run_tracker.py`; in/outflow options in `inoutflow.py`).

## Modules in `cbtv/`

Status: **ship** = used by both presets (`ship-v4`, and `bt` which adds `--coast-predict flow --exit-kill 1 --color-source track`); **default** = on by default in the tracker core;
**off** = implemented, off in `ship` (an option or a research path); **detector** = used only by that detector.

| module | role | status |
|---|---|---|
| `run_tracker.py` | the main loop: detection, motion prior, constellations, assignment, lifecycle, output | ship |
| `config.py` | command-line options of the tracker core (`build_parser`, `validate_args`) | ship |
| `gpu.py` | GPU/CPU backend: device init (`init_gpu`), array helpers; turns OpenCL off on `--use-gpu 0` | ship |
| `detectors.py` | detector registry: `available()`, `run_detector()` | ship |
| `ws_detector.py` | `ws_multi_vote` / `ws_vgl_log` adapter over the vendored detection modules | detector (default) |
| `csv_detector.py` | `csv` detector: reads your detections file | detector |
| `cellpose_detector.py` | `cellpose` detector | detector |
| `detection.py`, `exemplars.py`, `correlation.py`, `bayes.py`, `fp_memory.py`, `log_detect.py` | the `bayes_adapt` detector: exemplars, NCC correlation, Bayesian refinement, LoG | detector |
| `detection_filter.py`, `detect_utils.py`, `edge_fit.py` | detection filtering, de-duplication, sub-pixel edge-fit refinement | default |
| `polarity.py` | `--input-polarity`: bright-background video made to look like dark-background video | default (`auto`) |
| `piv.py` | dense image-correlation motion prior (`--mu-source piv`) | ship |
| `clusters.py` | colour constellations: coherence graph, cluster proposal and matching | ship |
| `color.py` | particle colours (CIELAB) and colour-distance matrices | ship |
| `match_ops.py` | structural colour-neighbourhood score (`--match-op graph-struct`) | ship |
| `cost_matrix.py` | the per-frame cost matrix (`compute_cost_matrix_gpu`, CPU and GPU paths) | ship |
| `assignment.py` | three-stage assignment: confident greedy, 2x2 blocks, Hungarian | ship |
| `tracks.py` | `Track`, the structure-of-arrays mirror `TrackArrays`, `KernelCache` | ship |
| `kernels.py`, `recovery.py` | image-space LAB tracking kernels (`--match-op kernel`) and the kernel-recovery pass for unmatched tracks (`--kernel-recovery`, default 1) | default |
| `inoutflow.py` | particles entering/leaving the view: region map, coasting, row tags (user guide) | `bt` uses `--coast-predict flow` and `--exit-kill 1`; the rest off |
| `field.py` | local affine deformation field (`--use-affine-field`) | off |
| `cascade.py` | cascade tier-1 exclusion pass (`--cascade-tier1`) | off |
| `pkt.py` | predictive-kernel early lock (`--opt-predictive-kernel`; has a known caching issue) | off |

Preset flags: `SHIP_CONFIG_V4` and `BT_CONFIG` in `run_cbtv.py` (`SHIP_CONFIG` aliases the default, `bt`).

## Detection subpackage (`cbtv/detect/`)

The code the shipped detectors need from the original research modules, moved under `cbtv/` with names
unchanged: `channels.py` (background subtraction, `build_channels`, the module-level
detection flags), `strategies.py` (`ws_multi_vote_gpu`, `ws_vgl_log_gpu`), `gpu_strategies.py` and
`gpu_backend.py` (CuPy when available, else NumPy/OpenCV). The research modules themselves are no longer
part of the repository.

## Debug hooks

- `CBTV_DUMP_C=<dir>`: write each frame's final cost matrix (`C_<frame>.npz`: `C`, `track_ids`, `det_idx`,
  `valid_pairs`) from both the CPU and GPU paths.
