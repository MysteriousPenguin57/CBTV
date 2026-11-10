"""Synthetic high-frame-rate clip with known GT, generated on the fly (nothing large is committed).

Particles start on a jittered disc packing and move with a differential rotation (shear) plus a slowly varying
per-particle velocity (Ornstein-Uhlenbeck), so neighbours move relative to each other and linking gets hard as the
step grows. Mean displacement ~``disp`` diameters per frame at full rate: the "tiny displacement" windows of route A
(<= ~0.05 d/frame). Detections = GT + Gaussian localisation noise, with dropouts and false positives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLOURS_BGR = [(40, 40, 220), (40, 200, 40), (220, 120, 30)]


def make(n: int = 120, frames: int = 240, size=(320, 320), r: float = 6.0, disp: float = 0.03,
         rel: float = 0.4, noise: float = 0.25, recall: float = 0.97, fp_rate: float = 0.01, seed: int = 0):
    """Returns (gt, detections, colours): gt frame,gt_id,x,y,visible; detections frame,x,y,r,score.
    rel = share of the displacement that is particle-specific (OU) rather than the shared rotation."""
    rng = np.random.default_rng(seed)
    W, H = size
    d = 2 * r
    c = np.array([W / 2, H / 2])
    R = 0.5 * min(W, H) - 1.5 * d
    sp = 1.9 * d                                           # jittered hexagonal packing inside the disc
    m = int(2 * R / sp) + 2
    ii, jj = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
    lat = np.stack([(ii + 0.5 * (jj % 2)) * sp, jj * sp * np.sqrt(3) / 2], -1).reshape(-1, 2)
    lat = lat[np.linalg.norm(lat, axis=1) <= R - 0.2 * d]
    lat = lat[rng.permutation(len(lat))[:n]]
    xy = c + lat + rng.uniform(-0.1 * d, 0.1 * d, lat.shape)
    n = len(xy)
    # shared differential rotation: speed (1-rel)*disp*d at radius R/2
    v_shared = (1 - rel) * disp * d
    omega0 = v_shared / (0.5 * R)
    sig = rel * disp * d
    v_ind = rng.normal(0, sig, (n, 2))
    theta = 0.98
    colours = rng.integers(0, 3, n)
    rows = []
    ids = np.arange(1, n + 1)
    for f in range(frames):
        rows.append(np.column_stack([np.full(n, f), ids, xy[:, 0], xy[:, 1], np.ones(n)]))
        rr = xy - c
        rad = np.linalg.norm(rr, axis=1)
        om = omega0 * (1.5 - rad / R)
        vel = np.stack([-rr[:, 1], rr[:, 0]], 1) * om[:, None]
        v_ind = theta * v_ind + np.sqrt(1 - theta ** 2) * rng.normal(0, sig, (n, 2))
        xy = xy + vel + v_ind
        out = np.linalg.norm(xy - c, axis=1) > R + d       # keep particles in the cell: reflect radially
        if out.any():
            u = (xy[out] - c) / np.linalg.norm(xy[out] - c, axis=1)[:, None]
            xy[out] = c + u * (R + d)
            v_ind[out] *= -1
    gt = pd.DataFrame(np.concatenate(rows), columns=["frame", "gt_id", "x", "y", "visible"])
    gt = gt.astype({"frame": int, "gt_id": int, "visible": int})
    keep = rng.random(len(gt)) < recall
    det = gt[keep][["frame", "x", "y"]].copy()
    det["x"] += rng.normal(0, noise, len(det))
    det["y"] += rng.normal(0, noise, len(det))
    n_fp = int(round(fp_rate * len(gt)))
    fp = pd.DataFrame({"frame": rng.integers(0, frames, n_fp), "x": rng.uniform(0, W, n_fp),
                       "y": rng.uniform(0, H, n_fp)})
    det = pd.concat([det, fp], ignore_index=True)
    det["r"] = r
    det["score"] = 1.0
    det = det.sort_values(["frame", "x"], kind="stable").reset_index(drop=True)
    return gt, det, colours


def render(gt: pd.DataFrame, colours, path, size=(320, 320), r: float = 6.0, fps: float = 500.0) -> None:
    """Write the GT as an FFV1 clip of coloured discs (for running an external tracker on the synthetic case)."""
    import cv2
    W, H = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"FFV1"), fps, (W, H))
    if not vw.isOpened():
        raise OSError("OpenCV cannot write FFV1 to %s" % path)
    for _, g in gt.groupby("frame", sort=True):
        img = np.full((H, W, 3), 25, np.uint8)
        for gid, x, y in zip(g["gt_id"], g["x"], g["y"], strict=True):
            cv2.circle(img, (int(round(x * 4)), int(round(y * 4))), int(round(r * 4)),
                       COLOURS_BGR[int(colours[gid - 1])], -1, lineType=cv2.LINE_AA, shift=2)
        vw.write(img)
    vw.release()
