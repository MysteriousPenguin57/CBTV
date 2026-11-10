"""Temporal decimation (offsets 0..k-1) and reversal of a detection set (+ optional lossless clip).

Decimation by step k at offset o keeps source frames f0+o, f0+o+k, ... and renumbers them 0, 1, ... (the
processed-frame index every linker and ``--detector csv`` expects). Together the k offsets use every source frame
exactly once. det_id is kept, so links found on any derived sequence are links between the same detections.
Reversal renumbers f1, f1-1, ..., f0 as 0, 1, ...; a tracker run on it gives the backward pass of the
forward-backward check. Derived clips are written FFV1 (lossless), so a tracker sees the decoded source pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tracksio import write_detections, write_manifest


def frame_range(d: pd.DataFrame, n_frames: int | None = None):
    """(f0, f1) inclusive. Frames without detections inside the range still count as frames."""
    f0 = int(d["frame"].min()) if len(d) else 0
    f1 = int(d["frame"].max()) if len(d) else -1
    if n_frames is not None:
        f0, f1 = 0, int(n_frames) - 1
    return f0, f1


def grid(f0: int, f1: int, k: int, offset: int):
    if not (k >= 1 and 0 <= offset < k):
        raise ValueError("need k >= 1 and 0 <= offset < k (got k=%s offset=%s)" % (k, offset))
    return list(range(f0 + offset, f1 + 1, k))


def derive(d: pd.DataFrame, src_frames) -> pd.DataFrame:
    """Detections on the given source frames, renumbered to their index in src_frames (det_id kept)."""
    pos = {int(f): i for i, f in enumerate(src_frames)}
    out = d[d["frame"].isin(pos)].copy()
    out["frame"] = out["frame"].map(pos).astype(int)
    return out.sort_values(["frame", "det_id"], kind="stable").reset_index(drop=True)


def decimate(d: pd.DataFrame, k: int, offset: int, n_frames: int | None = None):
    """(decimated detections, src_frames) for step k, offset o."""
    f0, f1 = frame_range(d, n_frames)
    src = grid(f0, f1, k, offset)
    return derive(d, src), src


def reverse(d: pd.DataFrame, n_frames: int | None = None):
    """(reversed detections, src_frames)."""
    f0, f1 = frame_range(d, n_frames)
    src = list(range(f1, f0 - 1, -1))
    return derive(d, src), src


def frames_xy(d: pd.DataFrame, n: int):
    """Per processed frame 0..n-1: (det_id array, xy (m,2), r (m,)) - empty frames included."""
    out = []
    by = {int(f): g for f, g in d.groupby("frame", sort=True)}
    for i in range(n):
        g = by.get(i)
        if g is None:
            out.append((np.zeros(0, int), np.zeros((0, 2)), np.zeros(0)))
        else:
            out.append((g["det_id"].to_numpy(), g[["x", "y"]].to_numpy(float), g["r"].to_numpy(float)))
    return out


def write_video(src_video, out_path, src_frames, fps: float | None = None) -> int:
    """Write the listed source frames, in that order, to an FFV1 avi. Returns frames written."""
    import cv2
    want = [int(f) for f in src_frames]
    need = set(want)
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise OSError("cannot open %s" % src_video)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    cache = {}
    i = 0
    last = max(need) if need else -1
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in need:
            cache[i] = fr
        i += 1
    cap.release()
    missing = [f for f in want if f not in cache]
    if missing:
        raise ValueError("%s has no frame(s) %s" % (src_video, missing[:5]))
    h, w = cache[want[0]].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"FFV1"), float(fps), (w, h))
    if not vw.isOpened():
        raise OSError("OpenCV cannot write FFV1 to %s" % out_path)
    for f in want:
        vw.write(cache[f])
    vw.release()
    return len(want)


def write_derived(d: pd.DataFrame, src_frames, out_csv, kind: str, k: int = 1, offset: int = 0,
                  video=None, source: str = "") -> dict:
    """Write a derived detections csv + manifest (+ clip next to it when video is given)."""
    out_csv = Path(out_csv)
    write_detections(out_csv, derive(d, src_frames))
    info = {"kind": kind, "k": int(k), "offset": int(offset), "src_frames": [int(f) for f in src_frames],
            "source_detections": source}
    if video is not None:
        clip = out_csv.with_suffix(".avi")
        write_video(video, clip, src_frames)
        info["clip"] = clip.name
    write_manifest(out_csv, info)
    return info


def decimate_all(d: pd.DataFrame, k: int, out_dir, video=None, n_frames: int | None = None, source: str = ""):
    """Offsets 0..k-1 -> out_dir/dets_k{k}_o{o}.csv (+ .seq.json manifest, + .avi clip with video)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f0, f1 = frame_range(d, n_frames)
    paths = []
    for o in range(k):
        p = out_dir / ("dets_k%d_o%d.csv" % (k, o))
        write_derived(d, grid(f0, f1, k, o), p, "decimate", k, o, video, source)
        paths.append(p)
    return paths
