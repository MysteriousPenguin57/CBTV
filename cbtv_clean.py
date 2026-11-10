#!/usr/bin/env python
"""cbtv_clean.py - clean a CBTV tracks.csv after the run.

Standalone (standard library only; does not import the tracker). Reads the
tracks.csv written by run_cbtv.py / cbtv.run_tracker and, when present, the
row tags (--row-tags 1: state, coast_n, region columns) and the
<stem>_meta.csv sidecar (birth/death reason per track).

Presets
-------
strict    (identity-safe positions only)
  1. drop every coasting row (states coast_frozen / coast_pred / occluded;
     without tags: every row with missed > 0);
  2. cut a track into two IDs at any step faster than --max-step bead
     diameters per frame (step / frame gap, so a gap is not a jump);
  3. drop segments with fewer than --min-matched detected rows;
  4. flag probable hand-offs: a segment born in the interior (or created by a
     cut) within --handoff-radius diameters and --handoff-frames frames of an
     exit-bound track's death. Flagged, never removed.
analysis  (keeps occluder coasts)
  keep coast_pred / occluded rows inside occluders (region 2), drop other
  coasting rows; no cuts, no length filter (override with the flags);
  hand-off flagging as in strict.

Outputs: <stem>_clean_<preset>.csv (the input columns + src_track_id,
flag, handoff_from) and <stem>_clean_<preset>_report.txt (rows and tracks
removed by rule).

Usage
-----
  python cbtv_clean.py results/tracks.csv --preset strict --d-p 19.6
  python cbtv_clean.py results/tracks.csv --preset analysis
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COAST_STATES = {"coast_frozen", "coast_pred", "occluded"}
OCCLUDER = 2
EXIT_DEATHS = {"exit_region", "exit_frame"}

PRESETS = {
    "strict":   dict(keep_occluder_coast=False, max_step=1.0, min_matched=3),
    "analysis": dict(keep_occluder_coast=True,  max_step=0.0, min_matched=0),
}


def missed_of(row):
    try:
        return int(float(row.get("missed") or 0))
    except ValueError:
        return 0


def is_coast(row):
    st = row.get("state")
    if st:
        return st in COAST_STATES
    return missed_of(row) > 0


def is_matched(row):
    return not is_coast(row)


def load_meta(path):
    meta = {}
    if path and Path(path).exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                meta[int(r["track_id"])] = r
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks_csv")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="strict")
    ap.add_argument("--meta", default=None,
                    help="tracks meta sidecar [<stem>_meta.csv next to the input]")
    ap.add_argument("--d-p", type=float, default=0.0,
                    help="bead diameter in px [2 x median r of detected rows]")
    ap.add_argument("--max-step", type=float, default=None,
                    help="cut tracks at steps faster than this, diameters/frame "
                         "(0 = no cuts) [strict 1.0, analysis 0]")
    ap.add_argument("--min-matched", type=int, default=None,
                    help="drop segments with fewer detected rows [strict 3, analysis 0]")
    ap.add_argument("--handoff-radius", type=float, default=1.0,
                    help="hand-off flag radius, diameters [1.0]")
    ap.add_argument("--handoff-frames", type=int, default=5,
                    help="hand-off flag window, frames after the death [5]")
    ap.add_argument("--handoff-deaths", choices=["exit", "any"], default="exit",
                    help="deaths that can hand off: exit (exit_region/exit_frame, needs "
                         "the meta sidecar; falls back to any without it) or any "
                         "non-end-of-video death [exit]")
    ap.add_argument("--out", default=None, help="cleaned CSV path")
    ap.add_argument("--report", default=None, help="report path")
    a = ap.parse_args(argv)

    src = Path(a.tracks_csv)
    cfg = dict(PRESETS[a.preset])
    if a.max_step is not None:
        cfg["max_step"] = a.max_step
    if a.min_matched is not None:
        cfg["min_matched"] = a.min_matched
    meta_path = Path(a.meta) if a.meta else src.with_name(src.stem + "_meta.csv")
    meta = load_meta(meta_path)
    out = Path(a.out) if a.out else src.with_name(f"{src.stem}_clean_{a.preset}.csv")
    rep = Path(a.report) if a.report else out.with_name(out.stem + "_report.txt")

    with open(src, newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        rows = list(rd)
    if not rows:
        sys.exit(f"[clean] {src} has no rows")
    tagged = "state" in fields
    lines = [f"cbtv_clean.py  preset={a.preset}  input={src}",
             f"row tags: {'yes' if tagged else 'NO (coasting = missed > 0; occluder coasts not identifiable)'}",
             f"meta sidecar: {meta_path if meta else 'none'}"]

    d_p = a.d_p
    if d_p <= 0:
        rs = [float(r["r"]) for r in rows if is_matched(r)] or [float(r["r"]) for r in rows]
        d_p = 2.0 * statistics.median(rs)
        lines.append(f"d_p estimated as 2 x median r = {d_p:.2f} px")
    else:
        lines.append(f"d_p = {d_p:.2f} px")

    n_rows_in = len(rows)
    ids_in = {int(r["track_id"]) for r in rows}
    f_first = min(int(r["frame"]) for r in rows)
    f_last = max(int(r["frame"]) for r in rows)

    # 1. coasting rows
    removed = defaultdict(int)
    kept = []
    for r in rows:
        if is_coast(r):
            st = r.get("state") or "coast(missed>0)"
            occ_ok = (cfg["keep_occluder_coast"] and tagged
                      and st in ("occluded", "coast_pred")
                      and str(r.get("region", "")).strip() == str(OCCLUDER))
            if not occ_ok:
                removed[f"coast row: {st}"] += 1
                continue
        kept.append(r)

    # 2. cut at fast steps
    by_tid = defaultdict(list)
    for r in kept:
        by_tid[int(r["track_id"])].append(r)
    next_id = max(ids_in) + 1
    segments = []          # (seg_id, src_id, is_first_segment, [rows])
    last_seg = {}          # src_id -> seg_id of its final segment
    n_cuts = 0
    for tid in sorted(by_tid):
        rs = sorted(by_tid[tid], key=lambda r: int(r["frame"]))
        cur, seg_id, first = [rs[0]], tid, True
        for prev, r in zip(rs, rs[1:], strict=False):
            gap = int(r["frame"]) - int(prev["frame"])
            step = math.hypot(float(r["x"]) - float(prev["x"]),
                              float(r["y"]) - float(prev["y"])) / max(1, gap)
            if cfg["max_step"] > 0 and step > cfg["max_step"] * d_p:
                segments.append((seg_id, tid, first, cur))
                n_cuts += 1
                cur, seg_id, first = [r], next_id, False
                next_id += 1
            else:
                cur.append(r)
        segments.append((seg_id, tid, first, cur))
        last_seg[tid] = seg_id

    # 3. short segments
    final = []
    for seg in segments:
        n_m = sum(1 for r in seg[3] if is_matched(r))
        if n_m < cfg["min_matched"]:
            removed[f"row in segment with < {cfg['min_matched']} detected rows"] += len(seg[3])
            removed["_segments_short"] += 1
            continue
        final.append(seg)

    # 4. hand-off flags
    use_exit_only = a.handoff_deaths == "exit" and bool(meta)
    deaths = []            # (frame, x, y, seg_id, reason)
    for seg_id, tid, first, rs in final:
        last = rs[-1]
        if last_seg[tid] == seg_id:
            reason = meta.get(tid, {}).get("death_reason", "") if meta else ""
            if not reason:
                reason = "end_of_video" if int(last["frame"]) == f_last else "unknown"
        else:
            reason = "cut"
        if reason == "end_of_video":
            continue
        if use_exit_only and reason not in EXIT_DEATHS:
            continue
        deaths.append((int(last["frame"]), float(last["x"]), float(last["y"]), seg_id, reason))
    flags = {}
    rad = a.handoff_radius * d_p
    for seg_id, tid, first, rs in final:
        b = rs[0]
        bf = int(b["frame"])
        if bf == f_first:
            continue
        if first:
            br = meta.get(tid, {}).get("birth_reason", "") if meta else ""
            if br and br != "interior":
                continue
        for (df, dx, dy, did, dr) in deaths:
            if did != seg_id and 0 < bf - df <= a.handoff_frames and \
                    math.hypot(float(b["x"]) - dx, float(b["y"]) - dy) <= rad:
                flags[seg_id] = (did, dr)
                break

    out_fields = fields + ["src_track_id", "flag", "handoff_from"]
    n_rows_out = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        final_rows = []
        for seg_id, tid, first, rs in final:
            fl = flags.get(seg_id)
            for r in rs:
                rr = dict(r)
                rr["src_track_id"] = tid
                rr["track_id"] = seg_id
                rr["flag"] = "handoff_suspect" if fl else ""
                rr["handoff_from"] = fl[0] if fl else ""
                final_rows.append(rr)
        final_rows.sort(key=lambda r: (int(r["frame"]), int(r["track_id"])))
        for rr in final_rows:
            w.writerow(rr)
        n_rows_out = len(final_rows)

    n_short = removed.pop("_segments_short", 0)
    lines += ["",
              f"rows  in {n_rows_in}  out {n_rows_out}  removed {n_rows_in - n_rows_out}",
              f"tracks in {len(ids_in)}  out {len(final)}  "
              f"(cuts +{n_cuts} segments, short segments dropped {n_short})",
              "", "rows removed by rule:"]
    lines += [f"  {k}: {v}" for k, v in sorted(removed.items())] or ["  none"]
    lines += ["", f"cut rule: step > {cfg['max_step']} d_p/frame "
                  f"({cfg['max_step'] * d_p:.1f} px/frame)" if cfg["max_step"] > 0
              else "cut rule: off",
              f"hand-off rule: born {'interior/cut' if meta else 'after the first frame'} within "
              f"{a.handoff_radius} d_p ({rad:.1f} px) and {a.handoff_frames} frames of "
              f"{'an exit' if use_exit_only else 'a non-end-of-video'} death",
              f"hand-off suspects flagged: {len(flags)} segments"]
    reasons = defaultdict(int)
    for did, dr in flags.values():
        reasons[dr] += 1
    if reasons:
        lines.append("  by death reason of the dying track: " +
                     ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    lines += ["", f"cleaned CSV: {out}"]
    rep.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
