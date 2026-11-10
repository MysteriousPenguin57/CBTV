# Example: impulsive collision of two particle blobs

Two dense blobs of 534 coloured particles collide in open space. The video is a rendering of a discrete-element
simulation (LAMMPS) made by the authors; it is one of the flows in the CBTV paper. The particles have three colours,
are about 13 px across, and the frame is 1024 x 1024 px.

| file | contents |
|---|---|
| `clip.mkv` | 101 frames: every 4th frame of the 401-frame simulation video (FFV1, lossless, 15 MB). Clip frame k is simulation frame 4k |
| `detections.csv` | the particle detections for each clip frame, `frame,x,y,r,score` (pixels; `frame` is the clip frame) |

Taking every 4th frame makes the particles move about 0.6 particle diameters between frames, which is where
identities start to become ambiguous for linkers that rely on position alone.

## Run it

From this folder, after installing CBTV (see the main README):

```
cbtv clip.mkv --detections-csv detections.csv --gpu 0
```

`--gpu 0` runs on the CPU. With a working CUDA GPU and the `[gpu]` extra, use `--gpu 1` instead, or leave the
option out to let CBTV choose. CPU and GPU give the same `tracks.csv`.

The run took 2 to 3.5 minutes on the CPU (107 s and 204 s in two runs on an Intel Core i7-14700KF) and 95 s on the
GPU (NVIDIA RTX 3060). Much of the time goes into writing the two output videos.

## What you get

CBTV writes into `results/`: `tracks.csv` (one row per particle per frame), `annotated.mp4` (each particle circled
in its track's colour, with its track ID), `overlay.mp4` (detections, live tracks and coasting tracks),
`summary.txt` and `cluster_log.csv`. The user guide (`docs/user_guide.md`) describes each file.

`summary.txt` should start with:

```
frames processed          : 101
unique tracks             : 538
track length median/mean  : 101 / 100.3 frames
tracks spanning full video: 530 (98.5%)
particles per frame median: 534
coasting rows (missed>0)  : 38 (0.1% of rows)
```

This is a good result: 530 of the 534 particles keep one identity through all 101 frames, and only four extra tracks
were started. Scored against the simulation's ground truth, 99.96% of the particle links from one frame to the next
were correct (11 identity swaps in about 53,400 links).

## Source and licence

The video and detections come from the authors' simulations and are distributed with CBTV under the same BSD
3-Clause licence (`LICENSE`). The detections were made with Cellpose and are the detections every tracker received in
the paper's comparison.
