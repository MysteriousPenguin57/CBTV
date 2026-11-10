#!/usr/bin/env python
"""render_states.py - QA overlay of a CBTV run by row state.

Draws every tracks.csv row on the input video, coloured by what the row is
(needs --row-tags 1 for the full picture; without tags a row is 'matched'
when missed == 0 and 'coast' otherwise):

  matched             green
  tentative_backfill  cyan     (detected; written when the birth was confirmed)
  coast_pred          orange   (coasting, position predicted)
  coast_frozen        grey     (coasting, position held)
  occluded            magenta  (coasting through an occluder)

With the <stem>_meta.csv sidecar, births (+) and deaths (x) are marked for
--marker-frames frames with their reason (B:entry_band, D:exit_frame, ...).
With --region-map the region labels are tinted (exit red, occluder magenta,
entry yellow) and, with --region-transform, the per-frame visible window:
area outside it is darkened. Region options mean exactly what they mean for
the tracker (see cbtv/inoutflow.py).

Examples
  python render_states.py --input stab.avi --tracks results/tracks.csv --out states.mp4
  python render_states.py --input stab.avi --tracks results/tracks.csv \\
      --region-map map.png --region-transform T.csv --region-transform-mode rig \\
      --png-dir qa_frames --frames 100:160
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STATE_BGR = {
    "matched": (60, 200, 60),
    "tentative_backfill": (230, 220, 40),
    "coast_pred": (0, 150, 255),
    "coast_frozen": (160, 160, 160),
    "occluded": (220, 60, 220),
    "coast": (160, 160, 160),
}
REGION_BGR = {1: (40, 40, 220), 2: (220, 60, 220), 3: (40, 220, 230)}
SHORT = {"first_frame": "first", "entry_band": "entry", "interior": "int",
         "after_occluder": "occ", "restart": "restart", "exit_region": "exit_reg",
         "exit_frame": "exit_frm", "miss_timeout": "timeout", "end_of_video": "end"}


def parse_frames(spec):
    if not spec:
        return None
    a, _, b = spec.partition(":")
    return int(a or 0), int(b) if b else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="the video the tracker ran on")
    ap.add_argument("--tracks", required=True, help="tracks.csv")
    ap.add_argument("--meta", default=None, help="[<stem>_meta.csv next to --tracks]")
    ap.add_argument("--out", default=None, help="output MP4 [<tracks stem>_states.mp4]")
    ap.add_argument("--png-dir", default=None, help="write PNG frames here instead of an MP4")
    ap.add_argument("--frames", default="", help="processed-frame range a:b (default all)")
    ap.add_argument("--frame-step", type=int, default=1, help="the run's --frame-step [1]")
    ap.add_argument("--label", choices=["id", "none"], default="id")
    ap.add_argument("--marker-frames", type=int, default=6,
                    help="frames a birth/death marker stays visible [6]")
    ap.add_argument("--region-map", default="")
    ap.add_argument("--region-transform", default="")
    ap.add_argument("--region-transform-mode", choices=["camera", "rig"], default="camera")
    ap.add_argument("--tint", type=float, default=0.25, help="region tint opacity [0.25]")
    a = ap.parse_args(argv)

    tracks_csv = Path(a.tracks)
    rows = defaultdict(list)
    with open(tracks_csv, newline="") as f:
        for r in csv.DictReader(f):
            st = r.get("state") or ("matched" if int(float(r["missed"])) == 0 else "coast")
            rows[int(r["frame"])].append((int(r["track_id"]), float(r["x"]), float(r["y"]),
                                          float(r["r"]), st))
    if not rows:
        sys.exit(f"[render] {tracks_csv} has no rows")
    births, deaths = defaultdict(list), defaultdict(list)
    meta = Path(a.meta) if a.meta else tracks_csv.with_name(tracks_csv.stem + "_meta.csv")
    first_pos, last_pos = {}, {}
    for f in sorted(rows):
        for tid, x, y, r, _ in rows[f]:
            first_pos.setdefault(tid, (f, x, y))
            last_pos[tid] = (f, x, y)
    if meta.exists():
        with open(meta, newline="") as f:
            for m in csv.DictReader(f):
                tid = int(m["track_id"])
                if tid in first_pos:
                    bf, bx, by = first_pos[tid]
                    births[bf].append((tid, bx, by, m["birth_reason"]))
                if tid in last_pos:
                    df, dx, dy = last_pos[tid]
                    deaths[df].append((tid, dx, dy, m["death_reason"]))

    region = None
    if a.region_map or a.region_transform:
        from cbtv.inoutflow import RegionMap
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        sys.exit(f"[render] cannot open {a.input}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if a.region_map or a.region_transform:
        region = RegionMap(W, H, a.region_map, a.region_transform, a.region_transform_mode)
        gy, gx = np.mgrid[0:H, 0:W]
        gx, gy = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)

    rng = parse_frames(a.frames)
    f_lo = rng[0] if rng else 0
    f_hi = (rng[1] if rng and rng[1] is not None else max(rows) + 1)
    wr = None
    png_dir = Path(a.png_dir) if a.png_dir else None
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
    else:
        out = Path(a.out) if a.out else tracks_csv.with_name(tracks_csv.stem + "_states.mp4")
        wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps / a.frame_step), (W, H))

    step = max(1, a.frame_step)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo * step)
    n_out = 0
    for proc in range(f_lo, f_hi):
        ok, img = cap.read()
        if not ok:
            break
        for _ in range(step - 1):
            cap.read()
        if region is not None:
            region.set_frame(proc * step)
            lab = region.labels(gx, gy).reshape(H, W)
            outside = (region.outside_dist(gx, gy) > 0).reshape(H, W)
            tint = img.copy()
            for k, col in REGION_BGR.items():
                tint[lab == k] = col
            tint[outside] = (0, 0, 90)
            mask = (lab > 0) | outside
            img[mask] = cv2.addWeighted(img, 1 - a.tint, tint, a.tint, 0)[mask]
        counts = defaultdict(int)
        for tid, x, y, r, st in rows.get(proc, []):
            col = STATE_BGR.get(st, (255, 255, 255))
            counts[st] += 1
            thick = 2 if st in ("matched", "tentative_backfill") else 1
            cv2.circle(img, (int(round(x)), int(round(y))), int(max(3, round(r))), col, thick,
                       cv2.LINE_AA)
            if a.label == "id":
                cv2.putText(img, str(tid), (int(x) + int(r) + 2, int(y) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        for k in range(a.marker_frames):
            for tid, x, y, why in births.get(proc - k, []):
                p = (int(round(x)), int(round(y)))
                cv2.drawMarker(img, p, (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
                if k == 0 or a.label == "id":
                    cv2.putText(img, "B:" + SHORT.get(why, why), (p[0] + 8, p[1] + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            for tid, x, y, why in deaths.get(proc - k, []):
                if why == "end_of_video":
                    continue
                p = (int(round(x)), int(round(y)))
                cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(img, "D:" + SHORT.get(why, why), (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)
        # legend
        y0 = 16
        cv2.rectangle(img, (4, 4), (210, 12 + 16 * 7), (0, 0, 0), -1)
        cv2.putText(img, f"frame {proc}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        for i, st in enumerate(("matched", "tentative_backfill", "coast_pred",
                                "coast_frozen", "occluded")):
            n = counts.get(st, 0) + (counts.get("coast", 0) if st == "coast_frozen" else 0)
            cv2.putText(img, f"{st} {n}", (10, y0 + 16 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, STATE_BGR[st], 1, cv2.LINE_AA)
        if png_dir:
            cv2.imwrite(str(png_dir / f"states_{proc:05d}.png"), img)
        else:
            wr.write(img)
        n_out += 1
    cap.release()
    if wr is not None:
        wr.release()
    print(f"[render] wrote {n_out} frames to {png_dir or out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
