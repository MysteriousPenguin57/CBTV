# Python API: `cbtv.track`

`cbtv.track(...)` is the `cbtv` command as a function. It returns the tracks as a pandas DataFrame, with the same
columns as `tracks.csv`. It runs the command's own code path: keyword arguments become command-line options, parsed
by the same parser and checked by the same code, and the tracker core runs with the preset's flags. A `track` call and
the equivalent `cbtv` command line therefore give byte-identical tracks.

```
pip install "cbtv[api]"        # cbtv + pandas (the DataFrame); add [gpu] for CUDA
```

## Track a video

```python
import cbtv

df = cbtv.track("my_experiment.mp4")                      # detector ws_multi_vote, preset bt, GPU if available
df = cbtv.track("my_experiment.mp4", r_min=4, r_max=8)    # particle radius range in px (user guide, "Tuning")
df = cbtv.track("clip.mp4", detections="detections.csv")  # your own frame,x,y,r[,score] detections
df = cbtv.track("clip.mp4", detections=my_dets_df)        # ... or a DataFrame with those columns
df = cbtv.track("clip.mp4", preset="ship-v4", gpu=False)  # the published v4 configuration, on CPU
```

The result has one row per track per processed frame: `frame, track_id, x, y, r, missed` (`missed` = consecutive
frames without a detection, 0 = matched). For example, track lengths:

```python
lengths = df.groupby("track_id").size()
```

## Arguments

| argument | default | meaning (the `cbtv` option) |
|---|---|---|
| `video` | required | input video path |
| `detections` | `None` | CSV path or DataFrame of detections; given -> the `csv` detector (`--detections-csv`) |
| `preset` | `"bt"` | `"bt"` or `"ship-v4"`; `cbtv.presets()` lists them with their tracker flags (`--preset`) |
| `gpu` | `"auto"` | `"auto"`, `True` (error if no CuPy/CUDA) or `False` (`--gpu auto/1/0`) |
| `detector` | `ws_multi_vote` (`csv` with `detections`) | detection preset (`--detector`) |
| `r_min`, `r_max` | 6, 10 | particle radius range in px (`--r-min`, `--r-max`) |
| `frame_step`, `max_frames` | 1, -1 | every Nth frame; stop after N frames, -1 = all |
| `input_polarity`, `seed`, `label`, `gpu_device`, `verbose` | as the command | `--input-polarity`, `--seed`, `--label`, `--gpu-device`, `--verbose` |
| `extra` | `""` | raw tracker-core flags, one string, e.g. `extra="--colour-ema 0.5"` (`--extra`) |
| `output_dir` | temporary | keep every output there (`tracks.csv`, `overlay.mp4`, `summary.txt`, ...); also in `df.attrs["output_dir"]` |
| `render` | `False` | also write `annotated.mp4` (the command always does) |

Bad inputs raise `ValueError` with the message the command prints after `[cbtv] error:`, an unknown keyword raises
`TypeError`, and a failure of the tracker core raises `RuntimeError`.

## Reproduce a golden

```python
import cbtv, pandas as pd
fx = "tests/fixtures/primary60/"
df = cbtv.track(fx + "clip.mp4", detections=fx + "detections.csv", preset="ship-v4", gpu=False)
assert df.equals(pd.read_csv(fx + "golden_gpu_tracks.csv"))   # bt: golden_gpu_tracks_bt.csv
```

This is `tests/test_api.py::test_primary60_dataframe_equals_golden`.

## API reference

::: cbtv.api.track

::: cbtv.api.presets
