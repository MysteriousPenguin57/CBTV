# Five-minute tutorial: track the bundled fixture

This walkthrough runs CBTV on `tests/fixtures/primary60`, a 60-frame, 1024 x 1024 clip of 534 coloured
particles (two dense blobs before they collide) shipped with the repository, together with its detection
CSV. You then check your result against the reference output. No GPU is needed.

## 1. Install

From the repository folder:

```
pip install .
cbtv --version
```

## 2. Run

```
cbtv tests/fixtures/primary60/clip.mp4 --detections-csv tests/fixtures/primary60/detections.csv --gpu 0 --preset ship-v4 --output-dir tutorial_out
```

- `--detections-csv` gives CBTV the fixture's detections, so the `csv` detector is used and the run is fully
  reproducible.
- `--preset ship-v4` is the configuration the reference output was made with. Without it CBTV uses its default,
  `bt`, which adds three edge and coasting flags (user guide, "Presets"); its reference output is
  `golden_gpu_tracks_bt.csv`.
- `--gpu 0` runs on CPU. Leave it out to use a GPU if one is available; the result is the same.

Expected runtime: about 2-2.5 minutes on CPU (141 s wall on the development machine, including the
annotated-video rendering); tracking alone took
116 s on CPU and 45-49 s on an RTX 3060 GPU (`tests/fixtures/MANIFEST.md`).

## 3. Look at the output

`tutorial_out/` now holds `tracks.csv`, `annotated.mp4`, `overlay.mp4` and `summary.txt`. The summary
should read:

```
frames processed          : 60
unique tracks             : 534
track length median/mean  : 60 / 60.0 frames
tracks spanning full video: 534 (100.0%)
particles per frame median: 534
coasting rows (missed>0)  : 0 (0.0% of rows)
```

Every one of the 534 particles keeps one identity through all 60 frames. Open `annotated.mp4` to see each
particle circled in its track's colour with its ID. The format of `tracks.csv` is described in the
[user guide](user_guide.md#output-trackscsv).

## 4. Verify against the reference

`tests/fixtures/primary60/golden_gpu_tracks.csv` is the reference output (GPU, `ship-v4`, the published configuration).
Your `tracks.csv` should be identical to it, byte for byte:

```
cmp tutorial_out/tracks.csv tests/fixtures/primary60/golden_gpu_tracks.csv
```

No output means identical. On Windows without `cmp`, use
`fc /b tutorial_out\tracks.csv tests\fixtures\primary60\golden_gpu_tracks.csv` ("no differences encountered").

`python tools/parity/diff_tracks.py tutorial_out/tracks.csv tests/fixtures/primary60/golden_gpu_tracks.csv` also
reports row counts and the first differing row.

## 5. Next steps

- Your own video: `cbtv my_experiment.mp4`, then adjust `--r-min/--r-max` (user guide, "Tuning").
- Your own detections: `cbtv my_experiment.mp4 --detections-csv my_detections.csv`.
- All options: [user guide](user_guide.md).
