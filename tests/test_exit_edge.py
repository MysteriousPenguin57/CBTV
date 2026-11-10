"""T038: --exit-kill must only end tracks that LEAVE the view; a particle resting at / just past the frame edge,
detected every frame, keeps one track (BT preset: --coast-predict flow --exit-kill 1 --color-source track).

Each test was committed failing before the fix."""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENCV_OPENCL_RUNTIME="disabled", PYTHONHASHSEED="0",
               PYTHONIOENCODING="utf-8")
S, R, N = 256, 7, 30
STATIC = [(60, 60), (196, 60), (60, 150), (196, 150)]          # interior particles that give the frame a motion prior
COLS = [(60, 200, 255), (255, 120, 40), (40, 255, 120), (200, 60, 200)]


def _run(tmp_path, movers):
    """movers: list of (colour, [(frame, x, y) ...]) drawn + detected; STATIC particles in every frame."""
    clip = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (S, S))
    rows = ["frame,x,y,r,score"]
    for f in range(N):
        img = np.zeros((S, S, 3), np.uint8)
        for (x, y), c in zip(STATIC, COLS, strict=True):
            cv2.circle(img, (x, y), R, c, -1)
            rows.append(f"{f},{x}.0,{y}.0,{R}.0,1.0")
        for c, path in movers:
            for ff, x, y in path:
                if ff == f:
                    cv2.circle(img, (int(round(x)), int(round(y))), R, c, -1)
                    rows.append(f"{f},{x:.3f},{y:.3f},{R}.0,1.0")
        vw.write(img)
    vw.release()
    det = tmp_path / "det.csv"
    det.write_text("\n".join(rows) + "\n")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "run_cbtv.py"), "--input", str(clip), "--detector", "csv",
                        "--detections-csv", str(det), "--gpu", "0", "--output-dir", str(out), "--r-min", "5",
                        "--r-max", "9", "--extra", "--csv-trust 1 --row-tags 1"],
                       cwd=str(tmp_path), env=CPU_ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return pd.read_csv(out / "tracks.csv"), pd.read_csv(out / "tracks_meta.csv")


def _rows_near(tr, path, tol=3.0):
    """The tracks.csv row nearest each detection of `path` (within tol px), one per frame."""
    got = []
    for f, x, y in path:
        fr = tr[tr.frame == f]
        if len(fr):
            d = np.hypot(fr.x - x, fr.y - y)
            if d.min() <= tol:
                got.append(fr.loc[d.idxmin()])
    return pd.DataFrame(got)


@pytest.mark.parametrize("inside", [0.6, -3.0], ids=["0.6px_inside", "3px_outside"])
def test_particle_resting_at_bottom_edge_keeps_one_track(tmp_path, inside):
    rng = np.random.default_rng(1)
    y0 = (S - 1) - inside                               # centre 0.6 px inside, or 3 px below the last row
    path = [(f, 128 + rng.uniform(-0.3, 0.3), y0 + rng.uniform(-0.3, 0.3)) for f in range(N)]
    tr, meta = _run(tmp_path, [((255, 255, 255), path)])
    got = _rows_near(tr, path)
    assert len(got) == N, "the edge particle is missing from %d of %d frames" % (N - len(got), N)
    assert got.track_id.nunique() == 1, "edge particle split into tracks %s" % sorted(got.track_id.unique())
    assert (got.missed == 0).all(), "edge particle coasted on frames %s" % got[got.missed > 0].frame.tolist()
    tid = int(got.track_id.iloc[0])
    assert meta.loc[meta.track_id == tid, "death_reason"].iloc[0] == "end_of_video"


def test_particle_leaving_right_edge_is_killed_and_entrant_gets_new_id(tmp_path):
    y = 100.0
    out_path = []
    for f in range(N):
        x = (S - 40) + 3.0 * f
        if x > (S - 1) + R:                             # detections stop once > 1 r outside
            break
        out_path.append((f, x, y))
    f_last = out_path[-1][0]
    f_in = f_last + 6                                   # later, a particle enters at the same place
    in_path = [(f, (S - 1) + 0.5 * R - 2.0 * (f - f_in), y) for f in range(f_in, N)]
    tr, meta = _run(tmp_path, [((255, 255, 255), out_path), ((255, 255, 255), in_path)])
    leaving = _rows_near(tr, out_path)
    assert leaving.track_id.nunique() == 1
    tid = int(leaving.track_id.iloc[0])
    m = meta[meta.track_id == tid].iloc[0]
    assert m.death_reason in ("exit_frame", "exit_region"), m.to_dict()
    assert m.death_frame <= f_last + 1, "track of the leaving particle lived until frame %d (last detection %d)" % (
        m.death_frame, f_last)
    entering = _rows_near(tr, in_path)
    assert len(entering), "the entering particle was not tracked"
    assert tid not in set(entering.track_id), "the entering particle inherited the exited track's id"
