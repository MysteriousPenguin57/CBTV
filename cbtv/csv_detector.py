"""External-CSV detector adapter: every tracker gets the same detections.

Reads per-frame detections from a shared CSV (``frame,x,y,r,score``,
processed-frame index — the schema the benchmark's other trackers read
via ``--detections-csv``) so the cbtv linker sees byte-identical detections
to the rest of the field. Also ~4 s/frame faster than the ``cellpose``
preset because the U-Net no longer runs in-loop.

Mirrors :func:`cbtv.cellpose_detector.detect_cellpose`: returns the
detection 5-tuple ``(dets, [], gray, chans, dbg)`` with the SAME
bg-subtract + ``build_channels`` chans so the downstream cost-matrix /
LAB-kernel path is byte-identical — detection provenance is the only thing
that changes.
"""
from __future__ import annotations

import csv

_CACHE_PATH = None
_CACHE_BY_FRAME = None


def _load(csv_path: str):
    global _CACHE_PATH, _CACHE_BY_FRAME
    if _CACHE_PATH == csv_path and _CACHE_BY_FRAME is not None:
        return _CACHE_BY_FRAME
    by_frame: dict = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            fr = int(float(r["frame"]))
            by_frame.setdefault(fr, []).append(
                (float(r["x"]), float(r["y"]), float(r.get("r") or 8.0),
                 float(r.get("score") or 1.0)))
    _CACHE_PATH, _CACHE_BY_FRAME = csv_path, by_frame
    return by_frame


def detect_csv(frame_bgr, args, current_frame_idx: int):
    """Return cbtv's detection 5-tuple from the shared CSV for this frame."""
    from .detect import channels as _db

    csv_path = getattr(args, "detections_csv", None)
    if not csv_path:
        raise ValueError("--detector csv requires --detections-csv")

    # Canonical channel set (parity with detect_ws / detect_cellpose).
    _db.SUBTRACT_BACKGROUND = True
    _db.USE_FIXED_BG = False
    try:
        bgr_input = _db.subtract_background(frame_bgr)
    except Exception:
        bgr_input = frame_bgr
    chans, gray = _db.build_channels(bgr_input)

    dets = _load(csv_path).get(int(current_frame_idx), [])
    dbg = {"method": "csv", "raw": len(dets), "bayes_added": 0,
           "csv": str(csv_path)}
    return list(dets), [], gray, chans, dbg
