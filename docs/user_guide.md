# CBTV user guide

This guide covers running CBTV, its options and presets, tuning for your own video, and reading the output.
Installation is described in [install.md](install.md).

## Quick start

```
cbtv my_experiment.mp4
```

With no other options, CBTV uses a GPU if one works (otherwise the CPU), detects particles with the `ws_multi_vote`
detector, tracks them with the default `bt` preset, and writes into `results/`:

| file | contents |
|---|---|
| `tracks.csv` | the trajectories, one row per particle per frame (format in "Output: `tracks.csv`" below) |
| `annotated.mp4` | every particle circled in a persistent per-track colour, with its track ID |
| `overlay.mp4` | the tracker's diagnostic overlay (green = detections, orange = live tracks, grey = coasting) |
| `summary.txt` | track counts, track lengths and coverage, with notes on how to read them |
| `cluster_log.csv` | per-frame statistics of the constellation stage |

The three options you are most likely to change:

```
cbtv my_experiment.mp4 --r-min 8 --r-max 14         # particle radius range in pixels (default 6-10)
cbtv my_experiment.mp4 --detections-csv dets.csv     # use your own detections (frame,x,y,r[,score])
cbtv my_experiment.mp4 --max-frames 50 --output-dir test   # quick check on the first 50 frames
```

On a new video, run a short test on 50 frames first to check the radius range and the detections before tracking the
whole video. Two complete examples, with videos, detections and the expected results, are in `examples/` (see their
README files). Without installing, `python run_cbtv.py my_experiment.mp4` from the repository folder does the same.

## Options

| option | default | meaning |
|---|---|---|
| (video) or `--input` | (required) | the video (anything OpenCV can read: mp4, avi, mkv, ...) |
| `--preset` | `bt` | tracking configuration (see "Presets") |
| `--output-dir` | `results` | output directory |
| `--detector` | `ws_multi_vote` (`csv` if `--detections-csv` is given) | detection method (see "Detectors") |
| `--detections-csv` | none | your own detections (`frame,x,y,r[,score]`), used with `--detector csv` |
| `--gpu` | `auto` | `auto` (CUDA if CuPy works, else CPU), `1` (GPU only), `0` (CPU only) |
| `--gpu-device` | `0` | CUDA device index |
| `--r-min` | `6` | minimum particle radius in pixels. This is the most important setting |
| `--r-max` | `10` | maximum particle radius in pixels |
| `--input-polarity` | `auto` | `auto` decides from the first frame; `dark` means particles brighter than the background (no preprocessing); `bright` means particles on a lit background (see "Bright background") |
| `--frame-step` | `1` | process every Nth frame |
| `--max-frames` | `-1` | stop after N frames (-1 = the whole video) |
| `--label` | `id` | `id` draws track IDs next to the circles; `none` draws circles only |
| `--verbose` | `0` | `1` prints per-frame diagnostics and timing, the tracker arguments, and full tracebacks for input errors |
| `--seed` | `0` | random seed; the same seed and inputs give the same `tracks.csv` |
| `--version` | | print the version and exit |
| `--extra` | `""` | further tracker options as one quoted string, for example `--extra "--colour-ema 0.5"` (see "Advanced options") |

`cbtv --help` prints the same list.

## Presets

`--preset` selects the tracking configuration.

| preset | what it is |
|---|---|
| `bt` (default) | `ship-v4` plus three options: `--coast-predict flow` (a track that loses its particle moves along the motion prior instead of standing still), `--exit-kill 1` (a track ends when it leaves the frame) and `--color-source track` (detections are compared with the colour each track stored, not with the current frame at the track's old position). The flags are `BT_CONFIG` in `run_cbtv.py`. |
| `ship-v4` | the configuration as first frozen for the paper: constellation matching, a cluster budget scaled with the number of particles, a displacement-gated dense (PIV) motion prior, and the structural colour-neighbourhood matcher. The flags are `SHIP_CONFIG_V4` in `run_cbtv.py`. |

All CBTV results in the paper use `bt`. On the paper's 71 benchmark cases, `bt` scored a mean HOTA of 0.7701, against
0.7367 for `ship-v4`. In the runs that selected it, no case was more than 0.002 worse in HOTA or PTR, and the gains
were on fast, crowded cases (for example, the 25 degree chute at step 4: HOTA 0.431 against 0.246). This choice was
made on the benchmark itself. It
was then checked by a rule fixed in advance on two flows that were never used in development (6 of 6 cases; mean HOTA
0.7477 against 0.6524).

## Detectors

Detection is a separate step from linking particle identities from frame to frame, and `--detector` chooses it.

| detector | when to use |
|---|---|
| `ws_multi_vote` (default) | general purpose: multi-channel watershed with voting |
| `csv` | your own detections from `--detections-csv` (chosen automatically when that option is given). Add `--extra "--csv-trust 1"` when your detections are already validated. Otherwise CBTV's image-quality filter still runs on them and, on imagery unlike its dark-background design case, can discard most of them (86% on one bright-background video) |
| `cellpose` | neural segmentation (needs the `[cellpose]` extra; slow without a GPU). Strongest on difficult experimental imagery and on dyes of different brightness |
| `bayes_adapt` | the original CBTV detector: Hough and Laplacian-of-Gaussian exemplars, normalised cross-correlation and Bayesian refinement |
| `ws_vgl_log` | watershed with a Laplacian-of-Gaussian variant; try it if `ws_multi_vote` misses dim particles (GPU only) |
| `log` | Laplacian of Gaussian only; robust for small or blurred particles |
| `single_frame`, `no_bayes` | diagnostic variants |

Giving `--detections-csv` together with a different `--detector` is an error.

The detections CSV has one row per detected particle: `frame,x,y,r[,score]`, where `frame` is the **processed** frame
index (0, 1, 2, ... after `--frame-step`), `x, y` the centre in pixels and `r` the radius in pixels.

**Particles of several colours that are not equally bright.** The `ws_*` detectors vote across the equalised grey,
V and L channels and keep only what they agree on. When the dyes differ in brightness (for example, red particles at a
grey level of about 85 against green and blue at about 175), the darker dye falls below the Otsu threshold in some
channels, those channels vote against it, and real particles are dropped without a warning. On a 1080 x 1920 frame
of about 800 touching particles in three colours, `ws_multi_vote` gave 427 detections and `cellpose` 799 on the same
preprocessed frame. In this case, prefer `cellpose`. If you stay with a `ws_*` detector, check `overlay.mp4` for a
whole colour class going undetected.

## Tuning for your video

In order of importance:

1. **`--r-min` and `--r-max`.** Measure a few particle radii in an image editor and set the range to bracket them.
   The defaults (6 to 10 px) suit particles of about 12 to 20 px diameter. A wrong radius range is the most common
   cause of poor detection.
   Settings given in pixels should be scaled with the image scale. For the paper's one flow rendered at twice the
   image scale, four were doubled: `--r-min 12 --r-max 20 --extra "--max-dist 40 --piv-ambiguity-disc 6"` instead
   of 6, 10, 20 and 3. Other pixel-valued settings, for example `--boundary-margin` and the kernel-recovery radii
   (`--recovery-search-radius`, `--recovery-max-move`), were left at their defaults there.
2. **`--detector`.** If detection looks weak in `overlay.mp4` (many particles without a green circle, or many false
   green circles), try the other detectors on a 50-frame test. On dense packings where particles touch, `cellpose` is
   usually the strongest.
3. **`--frame-step`.** If the frame rate is high relative to the motion, processing every 2nd to 4th frame speeds
   everything up. CBTV keeps identities at larger displacements per frame than linkers that rely on position alone,
   but identities are most stable when the median displacement per frame stays below about half the particle spacing.
4. **Advanced options (through `--extra`).** The tracker core has many more parameters: the cost weights
   `--w-dist/--w-color/--w-kernel`, the colour smoothing `--colour-ema` (0.3), the match gate `--match-gate-d` (1.1),
   the constellation budget `--cluster-cap-alpha` (0.324), the PIV settings, and others. The defaults are the
   benchmarked configuration; change them only with a reason. Among the PIV settings, `--piv-ambiguity-disc`
   (3 px) is the half-width of the square around the correlation peak that is excluded when the second peak is sought.
   Scale it with the image (6 at twice the image scale): a square narrower than the peak lets the peak's own flank
   count as a second peak, and the PIV confidence collapses.

**What your video needs.** Roughly circular particles in a small number of distinct colours (the colours carry the
identity information), reasonably steady illumination, and a particle radius of about 4 to 15 px with the default
settings. For larger particles, scale the pixel-valued settings as described in "Tuning for your video" above.

## Bright background (`--input-polarity`)

CBTV's detection was designed for bright particles on a dark background. The background subtraction removes the
most common colour, and the watershed detectors threshold equalised brightness channels. On a video of particles on
a lit tray or a lightbox, these steps select the background, and the detector returns almost no particles. The
background subtraction also removes particle colour (subtracting a cream background at BGR 178/186/186 from a red
particle at 40/40/190 leaves 0/0/12).

`--input-polarity auto` (the default) checks the most common brightness of the first frame. If the background is
bright, it sets the background to black before any other step sees the frame. Particles keep their own colour and
shading, which the constellation matcher uses.

The background is separated from the particles by **chromaticity** (each channel divided by B+G+R), not by
brightness. Chromaticity does not change with multiplicative shading, so the background reads the same whether it is
directly lit or in the shadow between two touching particles, and the gaps inside a dense pile are removed. A rule
based on brightness reports every shadow as a particle. The threshold is taken from the image by the triangle method,
which finds the shoulder of the background peak. Otsu's method does not work here, because with several particle
colours it splits between particle populations and keeps only the most saturated dye.

Requirements and limits:

- Particles must differ in chromaticity from the background. Coloured particles on a neutral tray are the intended
  case; grey particles on a grey background will not separate.
- Glossy particles have a bright, nearly neutral highlight that scores as background. It is filled with a
  structuring element scaled from `--r-max`, so set the radius range before judging the result.
- `--input-polarity dark` keeps the original path exactly (the module is not imported). Runs on dark-background
  video give byte-identical output with the option absent, `dark` or `auto`.
- `bright` always applies the matte, for the case where `auto` guesses wrong. Do not force it on dark-background
  video: chromaticity is undefined for nearly black pixels, so the matte keeps the background instead of removing
  it. `auto` does not apply it to such video.

## Particles entering and leaving the view

By default, a track that loses its particle **coasts frozen** at its last position for up to `--miss-tolerance` 8
frames (14 within 20 px of the frame edge). Near an edge or an occluder, this causes identity errors: the frozen track
stays where the particle left and takes the next particle that arrives there. The options below handle entry, exit
and occlusion. They are off in the tracker core (output then identical to the original tracker); the default `bt`
preset turns on `--coast-predict flow`, `--exit-kill 1` and `--color-source track`. Pass the others with
`--extra "..."`.

| option | default | meaning |
|---|---|---|
| `--coast-predict` | `freeze` | `velocity`: a coasting track moves by its own velocity in each missed frame (written to `tracks.csv`); `flow`: by the frame's motion prior (PIV in the shipped configuration), else by its velocity |
| `--coast-velocity-ema` | auto | the velocity is an exponential moving average of the observed steps (1.0 = last step only). Auto = 0.3 with `--occluder-coast`, else 1.0. Over a 30-frame occlusion the noise of the last step dominates (synthetic test: 31% re-captured at 1.0, 92% at 0.3) |
| `--exit-kill` | `0` | `1`: a track predicted into an exit region, or more than `--exit-margin` radii outside the visible area, ends in that frame and takes part in no matching, so it cannot capture an entering particle |
| `--exit-margin` | `1.0` | radii outside the visible area before `--exit-kill` applies |
| `--region-map` | none | grey PNG of labels in the reference frame: 0 view, 1 exit, 2 occluder, 3 entry band. Detections in exit regions (with `--exit-kill`) and in occluders (with `--occluder-coast`) are ignored |
| `--region-transform` | none | CSV `frame,dx,dy[,theta][,cx,cy]`, the per-frame transform from rig to camera, `cam = R(theta)(rig - c) + c + (dx, dy)` (theta in radians, c defaults to the image centre, frames in between interpolated) |
| `--region-transform-mode` | `camera` | `camera`: raw clip; the map moves with the rig and the frame edges are fixed. `rig`: clip stabilised to the rig; the map is fixed and the **visible window** moves (pixels whose camera position falls outside the frame count as outside the view) |
| `--occluder-coast` | `0` | N > 0: inside an occluder, a track coasts by its velocity and its misses are not counted for up to N frames; it re-captures only within `--occluder-gate` radii of its predicted point of re-emergence |
| `--occluder-gate` | `2.0` | re-capture gate after an occluder, in radii |
| `--confirm-hits` | `1` | K > 1: a new track is tentative until K consecutive matches, and a tentative track that misses is dropped. Confirmed tracks keep their first frames (output is delayed by K-1 frames) |
| `--confirm-hits-entry` | `0` | K for tracks born in entry bands (0 = `--confirm-hits`) |
| `--boundary-margin` / `--boundary-miss-bonus` | `20` / `6` | the edge band (px) and its extra miss tolerance |
| `--units d --d-p PX` | `px` | read the explicitly given `--max-dist`, `--boundary-margin`, `--recovery-search-radius` and `--recovery-max-move` in particle diameters |
| `--row-tags` | `0` | `1`: add `state, coast_n, region` columns and write `tracks_meta.csv` (see below) |
| `--color-source` | `frame` | `track`: the colour term uses the colour stored at the last match; `min`: the smaller of the two per pair (see "Fast particles" below) |

A typical run on a clip stabilised to the rig (stabilise with a **constant black** border, never by replicating the
border, because replicated pixels look like particles):

```
cbtv stab.avi --extra "--row-tags 1 --coast-predict velocity --coast-velocity-ema 0.3 --exit-kill 1
    --occluder-coast 30 --confirm-hits 2 --region-map map.png --region-transform transform.csv
    --region-transform-mode rig"
```

When the camera moves, prefer clips stabilised to the rig. A velocity measured in a moving camera frame contains the
camera's own motion, so long coasts fail on raw clips. In a synthetic test with an oscillating camera, 3% of the
particles behind an occluder were re-captured on the raw clip and 92% on the stabilised clip.

**Fast particles: `--color-source track`.** By default (`frame`), the colour term compares each detection with the
current frame sampled at the track's previous position. Once a particle moves more than about half a radius per frame,
that sample is mostly background, and true matches fail the colour gate. `--color-source track` compares with the
colour the track stored when it was last matched (the same statistic, smoothed with `--colour-ema`). It is on in the
default `bt` preset. In a synthetic test (stabilised clip, detections given), the fragmentation (tracks per particle)
changed as follows:

| motion (diameters per frame) | fragmentation with `frame` | with `track` |
|---|---|---|
| 0.1 | 1.41 | 1.15 |
| 0.3 | 16.2 | 1.44 |
| 0.6 | 30.1 | 2.40 |
| 1.0 | 21.9 | 3.18 |

`track` has a cost on real footage. Colour drift from frame to frame (lighting, compression) cancels when both sides
are sampled from the same frame, but not against a stored colour. On a real clip of slow, settled particles, `track`
increased fragmentation (356 tracks against 245 with `frame`). `--color-source min` takes the smaller distance per
pair and gave the fewest tracks on that clip (239). Test the options on a representative part of your video.

**Row tags.** With `--row-tags 1`, every row of `tracks.csv` says what it is. `state` is `matched`, `coast_frozen`,
`coast_pred` (coasting, position predicted), `occluded` (coasting inside an occluder) or `tentative_backfill` (a
detected row written when its track was confirmed); `coast_n` is the number of consecutive misses; `region` is the map
label at the position (1 outside the visible area). `tracks_meta.csv` gives per track `birth_frame, birth_reason`
(`first_frame`, `entry_band`, `after_occluder`, `restart`, `interior`), `death_frame, death_reason` (`exit_region`,
`exit_frame`, `miss_timeout`, `end_of_video`), `n_matched` and `n_coast`.

## Cleaning and checking a run

**Cleaning.** `cbtv_clean.py` (standard library only) removes coasting rows and probable identity hand-offs:

```
python cbtv_clean.py results/tracks.csv --preset strict --d-p 19.6
python cbtv_clean.py results/tracks.csv --preset analysis
```

`strict` drops every coasting row, splits a track into two IDs at any step faster than `--max-step` diameters per
frame (1.0), drops segments with fewer than `--min-matched` detected rows (3), and flags probable hand-offs (a track
born in the interior within `--handoff-radius` diameters and `--handoff-frames` frames of the end of a track that left
the view). `analysis` keeps coasting rows inside occluders and drops other coasting rows. Both write
`tracks_clean_<preset>.csv` (with `src_track_id`, `flag` and `handoff_from`) and a report of what each rule removed.
Without row tags, every row with `missed > 0` counts as coasting.

**Overlay by state.** `render_states.py` draws a run on its video: circles coloured by row state (matched green,
tentative_backfill cyan, coast_pred orange, coast_frozen grey, occluded magenta), birth (+) and death (x) markers with
their reasons, and optionally the region map and the visible window of each frame (same `--region-*` options as the
tracker):

```
python render_states.py --input stab.avi --tracks results/tracks.csv --out states.mp4 --region-map map.png
    --region-transform transform.csv --region-transform-mode rig
python render_states.py --input stab.avi --tracks results/tracks.csv --png-dir qa --frames 100:160
```

## Reading `summary.txt`

- `tracks spanning full video` close to your true particle count, and `unique tracks` not much larger, indicate a
  clean run.
- `unique tracks` much larger than the spanning count means fragmentation. It is usually caused by detection dropouts
  (fix the radius range, detector or lighting) or by displacements that are too large (reduce `--frame-step`, or
  increase the frame rate).
- A high `coasting rows (missed>0)` percentage means detection is struggling; check `overlay.mp4`.
- With `--row-tags 1`, the summary also breaks rows down by state and tracks by birth and death reason.

## Output: `tracks.csv`

Columns: `frame, track_id, x, y, r, missed`.

| column | meaning | units |
|---|---|---|
| `frame` | processed frame index: 0, 1, 2, ...; source frame = `frame` x `--frame-step` | frames |
| `track_id` | persistent identity of the particle, an integer >= 1 | - |
| `x`, `y` | position; origin at the top left, y downwards | pixels |
| `r` | radius | pixels |
| `missed` | **number** of consecutive processed frames without a matched detection: 0 = detected in this frame, 3 = the third frame in a row without one | frames |

Every live track has one row in every frame. A row with `missed > 0` is a **coasting row**: the particle was not
detected, and the position is the tracker's estimate (by default frozen at the last detected position; see
`--coast-predict` above). A track ends after `--miss-tolerance` (default 8) consecutive misses, with a few extra frames
allowed near the frame edge. To keep only measured positions, drop rows with `missed > 0`.

With `--extra "--row-tags 1"`, the file gains `state, coast_n, region` columns, and `tracks_meta.csv` is written.

## CPU and GPU

- `--gpu auto` (the default) uses CUDA through CuPy when it works, and otherwise runs on the CPU and prints a `[WARN]`
  line. `--gpu 0` forces the CPU (and switches off OpenCV's OpenCL). `--gpu 1` forces the GPU and stops with an
  error if CuPy does not work.
- **CPU and GPU give the same result.** With the same inputs and `--seed`, CPU and GPU runs write byte-identical
  `tracks.csv` on the `csv` detector path. This holds for the three regression fixtures (`primary60`: 32040 rows,
  534 tracks; `m05_30`: 16020 rows, 534 tracks; `synth_io`: 8903 rows, 182 tracks), which `tests/test_parity.py`
  and `tests/test_bt.py` check, and, with the pinned environment of [install.md](install.md), for all 71 benchmark
  cases of the paper on Windows.
- One option behaves differently: the older pixel-kernel appearance score (`--match-op kernel`) is GPU-only. Both
  presets use the graph-structural score, which is computed identically on both devices.
- Speed: the `primary60` fixture (60 frames, 534 particles, 1024 x 1024) took 45 to 49 s on an RTX 3060 GPU and 116 s
  on the CPU, about 0.8 and 1.9 s per frame, start-up included.

## Platforms

- **Windows and Linux on x86-64.** Tracks are byte-identical to the reference outputs
  (`tests/fixtures/*/golden_gpu_tracks*.csv`) on CPU and GPU, across AVX2 and AVX-512 CPUs. Linux with a GPU has
  not been tested.
- **macOS on Apple silicon (arm64).** Tracks differ numerically from the x86-64 reference outputs from the first
  frame, not only in track numbering: on `synth_io` with `ship-v4`, the frame-0 positions differ by up to 17.6 px. The
  detections are the same, but the positions and colours that CBTV refines from the decoded pixels are not. The cause
  has not been identified; it may be the video decoder (AVFoundation against FFmpeg) or the numerical libraries.
  macOS runs are reproducible on macOS but do not match the Windows and Linux numbers bit for bit. The size of the
  difference in tracking metrics has not been measured; `python tools/ci/platform_metrics.py out_metrics --run`
  reports it per fixture and preset.

## Advanced options: `--extra`

`cbtv` exposes the common options. Every other tracker option is passed to the tracker core with `--extra`, as one
quoted string; options given there override the preset:

```
cbtv my_experiment.mp4 --extra "--miss-tolerance 5 --coast-predict velocity"
```

`python -m cbtv.run_tracker --help` lists them all. Some of its help texts still refer to the research history of
the options (see "Known issues").

## Python API

`cbtv.track()` runs the same pipeline from Python and returns the tracks as a pandas DataFrame (it needs the `[api]`
extra). See [api_python.md](api_python.md).

## Troubleshooting

`cbtv` checks its inputs before tracking. A problem stops it with exit code 2 and one line starting with
`[cbtv] error:`. Add `--verbose 1` to see the full traceback.

| message starts with | cause and fix |
|---|---|
| `input video not found` | check the path |
| `cannot read video` | OpenCV cannot open or decode it; convert it to mp4 (H.264) or avi |
| `--gpu 1 but no working CuPy/CUDA was found` | `pip install "cbtv[gpu]"`, or run on the CPU with `--gpu 0` |
| `--detector csv needs --detections-csv FILE` | give the CSV, or choose another detector |
| `detections CSV not found` / `cannot parse detections CSV` | check the path; values must be numbers |
| `detections CSV ... is missing column(s)` | the CSV needs `frame,x,y,r` (and optionally `score`) |
| `no detections in the first 5 processed frames` | the detector found nothing at the start; check its radius range and threshold |
| `median detection radius ... is far outside --r-min / --r-max` | the radius range does not match your particles; use the suggested `--r-min/--r-max` |
| `--detections-csv only works with --detector csv` | drop `--detector` or `--detections-csv` |

The CSV checks (the last five rows) apply only to `--detector csv`.

Other symptoms:

- **Few or no detections with a built-in detector.** The radius range is wrong, or the background is brighter than
  the particles (try `--input-polarity bright`; see "Bright background"). Check `overlay.mp4`.
- **A whole colour class is missed with `ws_multi_vote`.** The dyes differ in brightness; use `cellpose` (see
  "Detectors").
- **Many short tracks near the frame edge.** Particles enter and leave the view; see "Particles entering and leaving
  the view".
- **Slow.** Install the `[gpu]` extra, or use `--frame-step` or `--max-frames` for a first look.
- **The output video does not play.** It uses the `mp4v` codec; re-encode it with ffmpeg.

## Limitations

- Colour carries the identity information. Particles of a single colour can be tracked, but the advantage of the
  constellation stage largely disappears, and CBTV then behaves like a linker that relies on position.
- Particles entering and leaving the view are handled (tracks start and end; see above), but heavy through-flow
  reduces the benefit of the constellations.
- By default, the colour gate compares the detection with the current frame sampled at the track's previous position.
  When particles move more than about half a radius per frame, use `--color-source track` (on in `bt`) and
  stabilise camera motion.
- Beyond about two to three particle diameters per frame, the range of CBTV's motion prior, trackers that compensate
  for bulk motion were more accurate in the paper's benchmark.
- When many particles go undetected, CBTV's advantage shrinks or turns into a small deficit.

## Known issues

- Some internal steps fall back without a warning when they fail on a frame: the background subtraction then uses
  the raw frame, the appearance kernel of a single track is skipped, a failed kernel recovery is skipped, and parts
  of the detection filter run on the CPU instead of the GPU. The published results were produced with this
  behaviour, so it was kept. Run with `--verbose 1` if a run behaves unexpectedly.
- The `ws_vgl_log` detector works only on a GPU; on the CPU it stops with an error.
- On macOS with Apple silicon, the tracks differ numerically from those on Windows and Linux (see "Platforms").
- Some help texts of the tracker core (`python -m cbtv.run_tracker --help`) refer to the research history of the
  options rather than describing them plainly.
