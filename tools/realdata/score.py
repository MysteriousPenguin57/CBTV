"""Score a tracker against a reference (consensus pseudo-GT or true GT), both as tracks frames on source frames.

Per evaluated frame, tracker points (observed rows only) are matched one-to-one to reference points within
``gate`` px (Hungarian). Then, for a tracker run at step k:
  pos_err      mean / p95 distance of matched points (px)
  k-links      pairs of reference points of one reference track on consecutive evaluated frames (t, t+k);
               the full-rate reference is thereby chained across the skipped frames
  id_err       switched / (correct + switched): of the k-links whose two ends are both matched, the fraction
               whose ends carry different tracker ids (the ID-switch rate per link)
  link_err     (switched + missed) / k-links
  idsw         CLEAR-MOT ID switches (a reference track's matched tracker id changes)
The reference may be incomplete (pseudo-GT covers only consensus links): tracker points without a reference point
are not counted against the tracker, so false positives are outside these metrics.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

FIELDS = ["ref_points", "matched", "pos_err", "pos_err_p95", "klinks", "correct", "switched", "missed",
          "id_err", "link_err", "idsw"]


def match_frame(a, b, gate):
    """Indices (i, j, dist) of a one-to-one match of points a to b within gate."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    c = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    cg = np.where(c <= gate, c, 1e9)
    ri, ci = linear_sum_assignment(cg)
    ok = cg[ri, ci] < 1e9
    return ri[ok], ci[ok], c[ri[ok], ci[ok]]


def counts(tracks: pd.DataFrame, ref: pd.DataFrame, frames, gate: float) -> dict:
    """Raw counts (summable over offsets) on the evaluated source frames `frames` (sorted, step k grid)."""
    frames = [int(f) for f in frames]
    nxt = {f: g for f, g in zip(frames[:-1], frames[1:], strict=True)}
    tb = {int(f): g for f, g in tracks.groupby("frame", sort=False)}
    rb = {int(f): g for f, g in ref.groupby("frame", sort=False)}
    match: dict = {}                     # (frame, ref_id) -> tracker id
    dists = []
    n_ref = 0
    for f in frames:
        r = rb.get(f)
        if r is None:
            continue
        n_ref += len(r)
        t = tb.get(f)
        if t is None:
            continue
        i, j, dd = match_frame(r[["x", "y"]].to_numpy(float), t[["x", "y"]].to_numpy(float), gate)
        rid = r["track_id"].to_numpy()[i]
        tid = t["track_id"].to_numpy()[j]
        for a, b in zip(rid, tid, strict=True):
            match[(f, int(a))] = int(b)
        dists.append(dd)
    correct = switched = missed = klinks = 0
    present = set(zip(ref["frame"].astype(int), ref["track_id"].astype(int), strict=True))
    for f, rid in present:
        g = nxt.get(f)
        if g is None or (g, rid) not in present:
            continue
        klinks += 1
        ta, tb_ = match.get((f, rid)), match.get((g, rid))
        if ta is None or tb_ is None:
            missed += 1
        elif ta == tb_:
            correct += 1
        else:
            switched += 1
    idsw = 0
    last: dict = {}
    for f, rid in sorted(match):
        tid = match[(f, rid)]
        if rid in last and last[rid] != tid:
            idsw += 1
        last[rid] = tid
    dd = np.concatenate(dists) if dists else np.zeros(0)
    return {"ref_points": n_ref, "matched": int(len(dd)), "dist_sum": float(dd.sum()), "dists": dd,
            "klinks": klinks, "correct": correct, "switched": switched, "missed": missed, "idsw": idsw}


def finish(c: dict) -> dict:
    dd = c["dists"]
    both = c["correct"] + c["switched"]
    return {"ref_points": c["ref_points"], "matched": c["matched"],
            "pos_err": float(dd.mean()) if len(dd) else float("nan"),
            "pos_err_p95": float(np.percentile(dd, 95)) if len(dd) else float("nan"),
            "klinks": c["klinks"], "correct": c["correct"], "switched": c["switched"], "missed": c["missed"],
            "id_err": c["switched"] / both if both else float("nan"),
            "link_err": (c["switched"] + c["missed"]) / c["klinks"] if c["klinks"] else float("nan"),
            "idsw": c["idsw"]}


def merge(cs) -> dict:
    out: dict[str, Any] = {"ref_points": 0, "matched": 0, "dist_sum": 0.0, "klinks": 0, "correct": 0, "switched": 0,
                           "missed": 0, "idsw": 0}
    for c in cs:
        for k in out:
            out[k] += c[k]
    out["dists"] = np.concatenate([c["dists"] for c in cs]) if cs else np.zeros(0)
    return out


def score(tracks: pd.DataFrame, ref: pd.DataFrame, frames, gate: float) -> dict:
    return finish(counts(tracks, ref, frames, gate))


def score_offsets(runs, ref: pd.DataFrame, gate: float) -> dict:
    """runs: [(tracks on source frames, evaluated source frames)] e.g. one per decimation offset; pooled."""
    return finish(merge([counts(t, ref, fr, gate) for t, fr in runs]))
