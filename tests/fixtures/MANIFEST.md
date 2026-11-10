# Test fixtures

Three short clips with detections and reference tracks, used by the regression tests (`tests/test_golden.py`,
`tests/test_bt.py`, `tests/test_parity.py`), and one detector reference (`detect_ws/`).

| fixture | source | frames | particles |
|---|---|---|---|
| `primary60` | the impulsive collision of two particle blobs (discrete-element simulation, 1024 x 1024 px), frames 0-59 | 60 | 534 |
| `m05_30` | a hopper flow (discrete-element simulation), frames 0-29 | 30 | 534 |
| `synth_io` | a synthetic in/outflow clip (particles enter and leave the view; detections are the synthetic ground truth with 0.3 px noise and 4% misses, not a detector) | 60 | 182 tracks |

The clips of `primary60` and `m05_30` are OpenCV `mp4v` re-encodes of the first frames of the source videos, and their
`detections.csv` files are the source detections (Cellpose) for those frames. The `synth_io` clip comes from a synthetic
clip generator in the authors' research code.

## How the reference tracks were made

All runs use the `csv` detector, `--frame-step 1 --r-min 6 --r-max 10`, and the preset flags in `tools/run_fixtures.py`:

```
python tools/run_fixtures.py gpu <outdir> [fixture ...] [--preset ship-v4|bt]
python tools/run_fixtures.py cpu <outdir> [fixture ...] [--preset ship-v4|bt]
```

- `golden_gpu_tracks.csv`: preset `ship-v4`, GPU (NVIDIA GeForce RTX 3060), run twice, byte-identical.
- `golden_gpu_tracks_bt.csv`: preset `bt` (`ship-v4` plus `--coast-predict flow --exit-kill 1 --color-source track`). On
  `primary60` and `m05_30` it equals the `ship-v4` reference byte for byte, because the extra flags change nothing on those
  short clips.
- `cpu_baseline_tracks.csv`: preset `ship-v4` on CPU (`--use-gpu 0`). It is byte-identical to the GPU reference.

Measured wall time with start-up included, `ship-v4`: `primary60` 45-49 s on the GPU and 116 s on CPU, `m05_30` 23-24 s
and 59 s, `synth_io` 12-13 s and 16 s.

## Checksums (sha256)

| file | bytes | sha256 |
|---|---|---|
| `primary60/clip.mp4` | 1072883 | d1542bebcf30429c9107ab8137b48af2dabb58acbd39b446734313f63500a7b3 |
| `primary60/detections.csv` | 1019958 | 4b94fe013f70694ee81fee06dba29aa3528a542abc3b26353ced348a3a14f4b3 |
| `primary60/golden_gpu_tracks.csv` | 1013489 | 93d69d5f25ba93b2dbae9fc1cada880e109db6d48c9b5304e6babaa57583040b |
| `primary60/golden_gpu_tracks_bt.csv` | 1013489 | 93d69d5f25ba93b2dbae9fc1cada880e109db6d48c9b5304e6babaa57583040b |
| `primary60/cpu_baseline_tracks.csv` | 1013489 | 93d69d5f25ba93b2dbae9fc1cada880e109db6d48c9b5304e6babaa57583040b |
| `m05_30/clip.mp4` | 787363 | 745f47514498ad9676632287351560dcb65f5478e955a191489df30cdcbaf3fd |
| `m05_30/detections.csv` | 507318 | 9957fa5812ed9a543de831bc9a3e7b737500251f1d06c7c774c7c264dee78673 |
| `m05_30/golden_gpu_tracks.csv` | 504089 | aa4633d06611fe9301d88625dcc5c02dc498860263e2f1b99c383a9ae02a88d3 |
| `m05_30/golden_gpu_tracks_bt.csv` | 504089 | aa4633d06611fe9301d88625dcc5c02dc498860263e2f1b99c383a9ae02a88d3 |
| `m05_30/cpu_baseline_tracks.csv` | 504089 | aa4633d06611fe9301d88625dcc5c02dc498860263e2f1b99c383a9ae02a88d3 |
| `synth_io/clip.avi` | 2025134 | a5eb528ccdb9387d3d26e057fcf5da8089cf3db9ed949b50d2e2c0b9dd2726f4 |
| `synth_io/detections.csv` | 215557 | c9aa1b37383d233ba74a3f9dd952df201c7e3fdc8a8e17af07059a9cbf8d4bd9 |
| `synth_io/gt.csv` | 621035 | 43d6f51e74b4fd1c11854db80ab92c69dd143eae1353a647ef03aa836a7cb558 |
| `synth_io/golden_gpu_tracks.csv` | 273429 | 09af6929d28675bf2669c8917c0920967b67788e51cd138297ff37018bd67a21 |
| `synth_io/golden_gpu_tracks_bt.csv` | 273113 | c227f38d5641cc12110ae37ef7d385326e86ccf0a38dd7dee7bba8c05cad2298 |
| `synth_io/cpu_baseline_tracks.csv` | 273429 | 09af6929d28675bf2669c8917c0920967b67788e51cd138297ff37018bd67a21 |
| `detect_ws/primary60_f0-4.csv` | 250216 | df7ee29edc285bab1cf082dd08c0a77fd7f947df9e24c81b893abb07c1b3da22 |

`detect_ws/primary60_f0-4.csv` holds the `ws_multi_vote` and `ws_vgl_log` detections on frames 0-4 of `primary60`, written by
`tools/make_detect_ws_fixture.py`.
