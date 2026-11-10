# Example: flow down a 25 degree inclined chute

534 coloured particles flow down a chute inclined at 25 degrees. The video is a rendering of a discrete-element
simulation (LAMMPS) made by the authors; it is one of the flows in the CBTV paper. The chute is periodic along the
flow, so a particle that leaves the view at the right-hand edge comes back in at the left-hand edge. The particles
have three colours, are about 12 px across, and the frame is 1024 x 1024 px.

| file | contents |
|---|---|
| `clip.mkv` | 120 frames: every 2nd frame of the first 239 frames of the simulation video (FFV1, lossless, 17 MB). Clip frame k is simulation frame 2k |
| `detections.csv` | the particle detections for each clip frame, `frame,x,y,r,score` (pixels; `frame` is the clip frame) |

Taking every 2nd frame makes the particles move about 1.2 particle diameters between frames, about 14 px, in a dense
bed where neighbouring particles move together.

## Run it

From this folder, after installing CBTV (see the main README):

```
cbtv clip.mkv --detections-csv detections.csv --gpu 0
```

`--gpu 0` runs on the CPU. With a working CUDA GPU and the `[gpu]` extra, use `--gpu 1` instead, or leave the
option out to let CBTV choose. CPU and GPU give the same `tracks.csv`.

The run took 2.5 to 3.5 minutes on the CPU (154 s and 211 s in two runs on an Intel Core i7-14700KF) and 125 s on
the GPU (NVIDIA RTX 3060).

## What you get

CBTV writes into `results/`: `tracks.csv` (one row per particle per frame), `annotated.mp4`, `overlay.mp4`,
`summary.txt` and `cluster_log.csv`. The user guide (`docs/user_guide.md`) describes each file.

`summary.txt` should start with:

```
frames processed          : 120
unique tracks             : 1506
track length median/mean  : 50 / 47.3 frames
tracks spanning full video: 0 (0.0%)
particles per frame median: 534
coasting rows (missed>0)  : 7125 (10.0% of rows)
```

No track spans the whole clip because every particle crosses the view in about 70 frames. A track ends when its
particle leaves at the right-hand edge, and the particle gets a new track when it comes back in at the left-hand edge,
because CBTV does not link identities across the periodic boundary. In this run, all 971 tracks that ended before the
last frame ended within 100 px of the right-hand edge, and all 972 tracks that started after the first frame started
within 100 px of the left-hand edge. No track ended inside the view. This is the result to expect. If you see tracks
ending in the middle of the view, check the detections and the radius range.

## Source and licence

The video and detections come from the authors' simulations and are distributed with CBTV under the same BSD
3-Clause licence (`LICENSE`). The detections were made with Cellpose and are the detections every tracker received in
the paper's comparison.
