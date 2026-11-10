#!/usr/bin/env python
"""Relaxation-PTV baseline CLI (C003 / BASELINES_SPEC section 1).

  python tracker_relax.py --detections-csv dets.csv --output-csv tracks.csv --colour-mode none

Input columns frame,x,y,r,score[,colour] (frame = processed index, px; colour in {0,1,2}, needed for
--colour-mode gate|weight). Output exactly frame,track_id,x,y,r,missed (track_id >= 1, missed = 0).
--R-n / --R-c are in units of d = 2 x median radius; --R-s-mult scales the default search radius.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relaxation_ptv import RelaxParams, track  # noqa: E402

COLUMNS = ["frame", "track_id", "x", "y", "r", "missed"]


def read_detections(path, need_colour):
    by_frame = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if need_colour and "colour" not in (rd.fieldnames or []):
            raise SystemExit("--colour-mode needs a 'colour' column in %s" % path)
        for row in rd:
            fr = int(float(row["frame"]))
            by_frame.setdefault(fr, []).append(
                (float(row["x"]), float(row["y"]), float(row.get("r") or 0.0),
                 int(float(row["colour"])) if need_colour else 0))
    frames = []
    for fr in sorted(by_frame):
        a = np.array(by_frame[fr], float).reshape(-1, 4)
        frames.append((fr, a[:, :2], a[:, 2], a[:, 3].astype(int) if need_colour else None))
    return frames


def write_tracks(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for fr, tid, x, y, r, missed in rows:
            w.writerow([fr, tid, "%.3f" % x, "%.3f" % y, "%.3f" % r, missed])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--colour-mode", choices=["none", "gate", "weight"], default="none")
    d = RelaxParams()
    ap.add_argument("--R-n", type=float, default=d.R_n, help="neighbour radius / d (default %(default)s)")
    ap.add_argument("--R-c", type=float, default=d.R_c, help="consistency tolerance / d (default %(default)s)")
    ap.add_argument("--N-it", type=int, default=d.N_it, help="iterations, max 20 (default %(default)s)")
    ap.add_argument("--R-s-mult", type=float, default=d.R_s_mult, help="search radius multiplier")
    ap.add_argument("--memory", type=int, default=d.memory, help="frames a lost track may coast")
    ap.add_argument("--spring", action="store_true", help="Okamoto spring weight instead of the hard R_c test")
    a = ap.parse_args(argv)
    if not 1 <= a.N_it <= 20:
        ap.error("--N-it must be in 1..20")
    params = RelaxParams(R_n=a.R_n, R_c=a.R_c, N_it=a.N_it, R_s_mult=a.R_s_mult, memory=a.memory,
                         spring=a.spring, colour_mode=a.colour_mode)
    frames = read_detections(a.detections_csv, a.colour_mode != "none")
    t0 = time.perf_counter()
    rows, _ = track(frames, params, log=print)
    dt = time.perf_counter() - t0
    write_tracks(a.output_csv, rows)
    print("[RELAX] linked in %.2fs (%.1f ms/frame)" % (dt, 1000.0 * dt / max(1, len(frames))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
