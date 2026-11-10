"""CSV I/O shared by the realdata tools.

Detections: frame,x,y,r[,score] (the schema every linker consumes). Each detection gets a ``det_id`` = its row
index after a stable sort by frame, so all CBTV-free linkers and every snapped external track talk about the same
detections. Tracks: frame,track_id,x,y,r,missed (CBTV / baselines output). Ground truth: frame,gt_id,x,y[,visible].

A *sequence manifest* (json next to a derived detections csv) lists ``src_frames``: processed frame i of the derived
sequence is source frame src_frames[i]. Decimated and reversed sequences both use it, so any tracker run on a
derived sequence maps back to source frames the same way.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["frame", "track_id", "x", "y", "r", "missed"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_detections(path, keep_det_id: bool = False) -> pd.DataFrame:
    """Detections with a stable det_id (row order after a stable sort by frame). keep_det_id keeps the det_id
    column of a derived csv (the ids of the source detections it was cut from) instead."""
    d = pd.read_csv(path)
    missing = {"frame", "x", "y"} - set(d.columns)
    if missing:
        raise ValueError("%s: detections need columns frame,x,y (missing %s)" % (path, sorted(missing)))
    if keep_det_id and "det_id" in d.columns:
        out = prepare_detections(d.drop(columns="det_id"))
        out["det_id"] = d["det_id"].to_numpy()[np.argsort(d["frame"].to_numpy(), kind="stable")]
        return out
    return prepare_detections(d)


def prepare_detections(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["frame"] = d["frame"].astype(float).astype(int)
    if "r" not in d.columns:
        d["r"] = 8.0
    if "score" not in d.columns:
        d["score"] = 1.0
    d = d.sort_values("frame", kind="stable").reset_index(drop=True)
    d["det_id"] = np.arange(len(d), dtype=int)
    return d


def write_detections(path, d: pd.DataFrame, keep_det_id: bool = True) -> None:
    cols = ["frame", "x", "y", "r", "score"] + (["det_id"] if keep_det_id and "det_id" in d.columns else [])
    out = d[cols].copy()
    out.to_csv(path, index=False, float_format="%.3f", lineterminator="\n")


def read_tracks(path, observed_only: bool = True) -> pd.DataFrame:
    """Tracks csv; observed_only drops coasting rows (missed > 0), which carry predicted, not detected, positions."""
    t = pd.read_csv(path)
    missing = {"frame", "track_id", "x", "y"} - set(t.columns)
    if missing:
        raise ValueError("%s: tracks need columns frame,track_id,x,y (missing %s)" % (path, sorted(missing)))
    if "missed" not in t.columns:
        t["missed"] = 0
    if "r" not in t.columns:
        t["r"] = 0.0
    t["frame"] = t["frame"].astype(int)
    t["track_id"] = t["track_id"].astype(int)
    if observed_only:
        t = t[t["missed"] == 0]
    return t.reset_index(drop=True)


def write_tracks(path, t: pd.DataFrame) -> None:
    out = t[TRACK_COLUMNS].copy()
    out["missed"] = out["missed"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    out.to_csv(path, index=False, float_format="%.3f", lineterminator="\n")


def read_gt(path, visible_only: bool = True) -> pd.DataFrame:
    """Ground truth as a tracks frame (track_id = gt_id). visible_only keeps rows with visible == 1."""
    g = pd.read_csv(path)
    if visible_only and "visible" in g.columns:
        g = g[g["visible"] == 1]
    out = pd.DataFrame({"frame": g["frame"].astype(int), "track_id": g["gt_id"].astype(int),
                        "x": g["x"].astype(float), "y": g["y"].astype(float)})
    out["r"] = 0.0
    out["missed"] = 0
    return out.reset_index(drop=True)


def remap_frames(t: pd.DataFrame, src_frames) -> pd.DataFrame:
    """Processed frame index -> source frame through a manifest's src_frames (rows outside the list are dropped)."""
    src = np.asarray(src_frames, int)
    t = t[(t["frame"] >= 0) & (t["frame"] < len(src))].copy()
    t["frame"] = src[t["frame"].to_numpy()]
    return t.reset_index(drop=True)


def manifest_path(csv_path) -> Path:
    p = Path(csv_path)
    return p.with_name(p.stem + ".seq.json")


def write_manifest(csv_path, info: dict) -> Path:
    mp = manifest_path(csv_path)
    mp.write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    return mp


def read_manifest(csv_path) -> dict:
    return json.loads(manifest_path(csv_path).read_text())


def frames_of(d: pd.DataFrame):
    """Sorted unique frames and {frame: det_id array} for a detections frame."""
    groups = {int(f): g["det_id"].to_numpy() for f, g in d.groupby("frame", sort=True)}
    return sorted(groups), groups
