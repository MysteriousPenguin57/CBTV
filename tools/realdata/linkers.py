"""CBTV-free reference linkers and the det-level link representation.

A *link set* is ``succ``: an int array over det_id, succ[a] = b when detection a is linked to detection b on the next
processed frame of the sequence, -1 otherwise. Every linker here is built only from the detections (no CBTV code is
imported), so a reference built from them is independent of CBTV:

  relax  baselines/relaxation (Baek & Lee / Ohmi & Li relaxation PTV), memory 0
  nn     drift-compensated mutual nearest neighbour with a gate and a ratio test (the high-fps linker)
  lap    drift-compensated global assignment (Hungarian) within a gate

External tracker output (CBTV, trackpy, GOTH, ...) enters as a tracks csv snapped onto the detections
(``tracks_to_succ``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .decimate import frames_xy

ROOT = Path(__file__).resolve().parents[2]
BUILTIN = ("relax", "nn", "lap")


def _relaxation():
    p = str(ROOT / "baselines" / "relaxation")
    if p not in sys.path:
        sys.path.insert(0, p)
    import relaxation_ptv
    return relaxation_ptv


def diameter(d: pd.DataFrame) -> float:
    """d = 2 x median detection radius (px)."""
    return 2.0 * float(np.median(d["r"])) if len(d) else 16.0


def _mutual_nn(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    _, ja = cKDTree(b).query(a)
    _, ib = cKDTree(a).query(b)
    i = np.arange(len(a))
    keep = ib[ja] == i
    return i[keep], ja[keep]


def drift(a, b, iters: int = 2):
    """Global shift estimate: median displacement of mutual nearest neighbours, refined once."""
    s = np.zeros(2)
    for _ in range(iters):
        i, j = _mutual_nn(a + s, b)
        if len(i) == 0:
            break
        s = np.median(b[j] - a[i], axis=0)
    return s


def pair_nn(a, b, dpx, gate=0.5, ratio=0.7):
    """Mutual NN after drift compensation; accept if dist <= gate*d and nearest < ratio x second nearest
    (both directions)."""
    if len(a) == 0 or len(b) == 0:
        return []
    a2 = a + drift(a, b)
    kb = min(2, len(b))
    ka = min(2, len(a))
    da, ja = cKDTree(b).query(a2, k=kb)
    db, ib = cKDTree(a2).query(b, k=ka)
    da, ja = da.reshape(len(a), kb), ja.reshape(len(a), kb)
    db, ib = db.reshape(len(b), ka), ib.reshape(len(b), ka)
    out = []
    for i in range(len(a)):
        j = int(ja[i, 0])
        if int(ib[j, 0]) != i or da[i, 0] > gate * dpx:
            continue
        if kb > 1 and da[i, 0] >= ratio * da[i, 1]:
            continue
        if ka > 1 and db[j, 0] >= ratio * db[j, 1]:
            continue
        out.append((i, j))
    return out


def pair_lap(a, b, dpx, gate=1.0):
    """Hungarian assignment on drift-compensated distance; pairs farther than gate*d are dropped."""
    if len(a) == 0 or len(b) == 0:
        return []
    a2 = a + drift(a, b)
    c = np.linalg.norm(a2[:, None, :] - b[None, :, :], axis=2)
    big = 1e6
    cg = np.where(c <= gate * dpx, c, big)
    ri, ci = linear_sum_assignment(cg)
    return [(int(i), int(j)) for i, j in zip(ri, ci, strict=True) if cg[i, j] < big]


def link_sequence(name: str, dseq: pd.DataFrame, n: int, n_det: int, params: dict | None = None) -> np.ndarray:
    """Link a (derived) detection sequence with processed frames 0..n-1; returns succ over det_id (len n_det)."""
    params = dict(params or {})
    succ = np.full(n_det, -1, int)
    fr = frames_xy(dseq, n)
    dpx = diameter(dseq)
    if name == "relax":
        rp = _relaxation()
        rparams = rp.RelaxParams(memory=0, **{k: v for k, v in params.items() if hasattr(rp.RelaxParams, k)})
        rows, _ = rp.track([(i, xy, r, None) for i, (_, xy, r) in enumerate(fr)], rparams)
        tids: dict = {}
        for row in rows:
            tids.setdefault(int(row[0]), []).append(int(row[1]))
        for i in range(n - 1):
            ta, tb = tids.get(i, []), tids.get(i + 1, [])
            if len(ta) != len(fr[i][0]) or len(tb) != len(fr[i + 1][0]):
                if len(fr[i][0]) and len(fr[i + 1][0]):
                    raise RuntimeError("relaxation rows do not follow detection order at frame %d" % i)
                continue
            pos = {t: j for j, t in enumerate(tb)}
            for j, t in enumerate(ta):
                if t in pos:
                    succ[fr[i][0][j]] = fr[i + 1][0][pos[t]]
        return succ
    if name == "nn":
        fn = lambda a, b: pair_nn(a, b, dpx, params.get("gate", 0.5), params.get("ratio", 0.7))  # noqa: E731
    elif name == "lap":
        fn = lambda a, b: pair_lap(a, b, dpx, params.get("gate", 1.0))  # noqa: E731
    else:
        raise ValueError("unknown builtin linker %r (have %s)" % (name, ", ".join(BUILTIN)))
    for i in range(n - 1):
        ida, xya, _ = fr[i]
        idb, xyb, _ = fr[i + 1]
        for p, q in fn(xya, xyb):
            succ[ida[p]] = idb[q]
    return succ


def link_rows(succ: np.ndarray, d: pd.DataFrame, src_frames) -> pd.DataFrame:
    """Tracks frame (source frames) from a link set: chains of succ restricted to dets on src_frames."""
    on = d[d["frame"].isin(set(int(f) for f in src_frames))]
    ids = on["det_id"].to_numpy()
    has_pred = np.zeros(len(succ), bool)
    s = succ[ids]
    has_pred[s[s >= 0]] = True
    tid = np.full(len(succ), -1, int)
    nxt = 1
    for a in ids:                       # det_id order is frame order, so heads come before their successors
        if tid[a] < 0:
            tid[a] = nxt
            nxt += 1
        b = succ[a]
        if b >= 0:
            tid[b] = tid[a]
    out = on[["frame", "x", "y", "r"]].copy()
    out["track_id"] = tid[ids]
    out["missed"] = 0
    return out.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def snap(points: pd.DataFrame, d: pd.DataFrame, tol: float) -> np.ndarray:
    """det_id per point (source frames), -1 if no detection within tol px (one-to-one per frame, Hungarian)."""
    out = np.full(len(points), -1, int)
    pts = points.reset_index(drop=True)
    by = {int(f): g for f, g in d.groupby("frame", sort=False)}
    for f, g in pts.groupby("frame", sort=False):
        dg = by.get(int(f))
        if dg is None:
            continue
        a = g[["x", "y"]].to_numpy(float)
        b = dg[["x", "y"]].to_numpy(float)
        c = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        cg = np.where(c <= tol, c, 1e6)
        ri, ci = linear_sum_assignment(cg)
        ok = cg[ri, ci] < 1e6
        out[g.index.to_numpy()[ri[ok]]] = dg["det_id"].to_numpy()[ci[ok]]
    return out


def tracks_to_succ(t: pd.DataFrame, d: pd.DataFrame, src_frames, tol: float) -> tuple[np.ndarray, dict]:
    """Snap a tracks frame (already in source frames) onto detections; consecutive points of a track on
    consecutive entries of src_frames become links. Returns (succ, info)."""
    succ = np.full(len(d), -1, int)
    pos = {int(f): i for i, f in enumerate(src_frames)}
    t = t[t["frame"].isin(pos)].reset_index(drop=True)
    det = snap(t, d, tol)
    t = t.assign(_det=det, _i=t["frame"].map(pos)).sort_values(["track_id", "_i"], kind="stable")
    tid = t["track_id"].to_numpy()
    di = t["_det"].to_numpy()
    ii = t["_i"].to_numpy()
    m = (tid[:-1] == tid[1:]) & (ii[1:] == ii[:-1] + 1) & (di[:-1] >= 0) & (di[1:] >= 0)
    succ[di[:-1][m]] = di[1:][m]
    return succ, {"points": int(len(t)), "snapped": int((det >= 0).sum()), "links": int(m.sum())}
