"""Cellpose U-Net (CBTV-U) detector adapter for the cbtv linker.

Phase 5 detection-matched Group-M: the dense granular corpus collapsed under the
default Hough/LoG detector (F1 0.6-0.78), which confounded both the "detection
ceiling" and the cbtv-vs-GOTH comparison. This routes the cbtv linker through
Cellpose so detection can be held fixed at a strong detector.

Mirrors :func:`cbtv.ws_detector.detect_ws`: returns cbtv's detection 5-tuple
``(dets, [], gray, chans, dbg)`` with ``dets = [(x, y, r, score)]``. Cellpose
runs on the raw BGR->gray frame (matching the proven ``_cellpose_swap`` path);
``chans`` are built via ``cbtv.detect.channels.build_channels`` after bg-subtract
so the downstream cost-matrix / LAB-kernel path is byte-identical to the
ws_multi_vote preset. Detection is therefore the only thing that changes.

Cellpose 4.x: ``CellposeModel`` default model is ``cpsam`` (Cellpose-SAM); cached
module-global (load ~2s). Diameter = ``2*rmax+1`` (fixed; do NOT auto-estimate
per dense frame -- erratic). Wired as the ``cellpose`` preset in
:mod:`cbtv.detectors`; the default ``bayes_adapt`` path is untouched.
"""
from __future__ import annotations

import math

import numpy as np

_MODEL = None


def _get_model(gpu: bool = True):
    """Lazy-load + cache the Cellpose-SAM model (load is ~2s)."""
    global _MODEL
    if _MODEL is None:
        from cellpose import models  # deferred so import never fails w/o cellpose
        _MODEL = models.CellposeModel(gpu=gpu)
    return _MODEL


def _masks_to_dets(masks, score_map=None):
    """Cellpose label image -> [(x, y, r_equiv, score)]; r from mask area."""
    from scipy import ndimage as ndi
    if masks is None:
        return []
    masks = np.asarray(masks)
    if masks.size == 0 or masks.max() == 0:
        return []
    n = int(masks.max())
    labels = np.arange(1, n + 1)
    cents = ndi.center_of_mass(masks > 0, masks, labels)
    areas = ndi.sum_labels(np.ones_like(masks, dtype=np.float32), masks, labels)
    scores = ndi.mean(score_map, masks, labels) if score_map is not None else [1.0] * n
    out = []
    for (cy, cx), a, sc in zip(cents, areas, scores, strict=False):
        if a <= 0:
            continue
        r = float((a / math.pi) ** 0.5)
        try:
            s = 1.0 / (1.0 + math.exp(-float(sc)))  # squash cellprob logits -> 0..1
        except (TypeError, ValueError, OverflowError):
            s = 0.5
        out.append((float(cx), float(cy), r, s))
    return out


def detect_cellpose(frame_bgr, r_min, r_max):
    """Run Cellpose, return cbtv's detection 5-tuple (dets, [], gray, chans, dbg)."""
    import cv2

    from .detect import channels as _db

    # Canonical channel set for the downstream cost matrix (parity with detect_ws).
    _db.SUBTRACT_BACKGROUND = True
    _db.USE_FIXED_BG = False
    bgr_input = frame_bgr
    try:
        bgr_input = _db.subtract_background(frame_bgr)
    except Exception:
        bgr_input = frame_bgr
    chans, gray = _db.build_channels(bgr_input)

    # Cellpose runs on the raw BGR->gray intensity (proven path in _cellpose_swap).
    gray_cp = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    diameter = float(2 * int(r_max) + 1)
    model = _get_model(gpu=True)
    masks = None
    score_map = None
    try:
        res = model.eval(gray_cp, diameter=diameter)
        masks = res[0] if isinstance(res, tuple) else res
        if isinstance(res, tuple) and len(res) >= 2 and res[1] is not None:
            try:
                if len(res[1]) >= 3:
                    score_map = np.asarray(res[1][2])
            except Exception:
                score_map = None
    except Exception as e:  # noqa: BLE001
        print(f"[cellpose] eval raised: {e}")

    dets = _masks_to_dets(masks, score_map=score_map)
    dbg = {"method": "cellpose_cpsam", "raw": len(dets), "bayes_added": 0,
           "diameter": float(diameter)}
    return dets, [], gray, chans, dbg
