"""Phase 5 Speed Stage 3 — graph-struct matcher (kernel-free structural score).

The LAB-neighbourhood kernel + NCC is the +320 ms kernel-BUILD hotspot and only
applies when w_kernel>0. This module computes a structural compatibility score
for a (track -> candidate-detection) pair WITHOUT any pixel kernel: take the
track's neighbour star-graph in frame n (relative offsets + colours), anchor it
at the candidate detection in n+1, deform the offsets by the local PIV/field
transform B, and score how well the predicted neighbour positions line up with
the DETECTED neighbours in n+1 -- reusing clusters.score_hypothesis (the gated
one-to-one correspondence) + clusters.cluster_confidence (the graded [0,1] score
with the colour moat + lattice-shift guard). Viable now because detection F1 is
~0.99 on the granular corpus, so neighbours are reliably present in n+1 (the
original blocker).

Returns a score in [0,1] usable IN the cost matrix in place of / blended with
the kernel-NCC term. --match-op kernel (default) keeps the shipped pixel kernel
(byte-identical); --match-op graph-struct routes here. This file is the Stage-3
core; the cost_matrix wiring + GPU bake-off are the follow-on.
"""
from __future__ import annotations

import numpy as np

from .clusters import cluster_confidence, score_hypothesis
from .field import apply_affine


def predict_neighbours(anchor_xy, neighbour_offsets, B=None, alpha=0.0):
    """Predicted absolute neighbour positions in n+1: anchor + (I+alpha*B)*offset.

    anchor_xy: (2,) candidate detection position in n+1.
    neighbour_offsets: (k,2) neighbour positions MINUS the track position in n.
    B: (2,2) local field Jacobian or None; alpha in [0,1] field confidence.
    With alpha=0 (or B None) this is a pure translation (offsets preserved).
    """
    off = np.asarray(neighbour_offsets, dtype=np.float64)
    if B is None or alpha == 0.0:
        deformed = off
    else:
        deformed = np.stack([apply_affine(B, off[i], alpha) for i in range(off.shape[0])])
    return np.asarray(anchor_xy, dtype=np.float64)[None, :] + deformed


def graph_struct_score(anchor_xy, neighbour_offsets, neighbour_radii,
                       neighbour_colors, det_xy, det_colors, free_dets,
                       *, B=None, alpha=0.0, gate_px=8.0, colour_max=60.0,
                       lambda_colour=0.5):
    """Structural compatibility in [0,1] for anchoring a track's neighbour
    star-graph at anchor_xy (a candidate detection) in frame n+1.

    Reuses score_hypothesis (gated one-to-one neighbour correspondence) +
    cluster_confidence (inlier fraction x residual x colour-moat margin). A
    correct anchor lands the whole star on real detections (high n_matched,
    low residual, finite margins -> high score); a one-spacing-wrong anchor
    leaves neighbours unsupported (low n_matched -> low score)."""
    pred = predict_neighbours(anchor_xy, neighbour_offsets, B=B, alpha=alpha)
    k = pred.shape[0]
    if k == 0:
        return 0.0, {"n_matched": 0}
    sc = score_hypothesis(
        pred, np.asarray(neighbour_radii, dtype=np.float64),
        np.asarray(neighbour_colors, dtype=np.float64),
        np.asarray(det_xy, dtype=np.float64),
        np.asarray(det_colors, dtype=np.float64),
        np.asarray(free_dets, dtype=bool),
        gate_px=gate_px, colour_max=colour_max, lambda_colour=lambda_colour)
    inl = sc["assign"] >= 0
    n_in = int(inl.sum())
    mean_res = float(np.mean(sc["residual"][inl])) if n_in else float("inf")
    fin_marg = sc["margin"][inl]
    fin_marg = fin_marg[np.isfinite(fin_marg)]
    mean_marg = float(np.mean(fin_marg)) if fin_marg.size else float("inf")
    conf = cluster_confidence(k, n_in, mean_res, mean_marg, gate_px=gate_px)
    return float(conf), {"n_matched": sc["n_matched"], "mean_residual": mean_res,
                         "mean_margin": mean_marg}


def build_graph_struct_data(track_ref_xy, track_ids, neighbor_lists,
                            prev_det_xy, prev_det_col, affine_by_id, kmax=8):
    """Per-track padded star-graph arrays for the cost-matrix graph-struct path.

    track_ref_xy (M,2): each track's reference (previous) position; track_ids (M,);
    neighbor_lists {track_id: [indices into prev_det_*]}; prev_det_xy (P,2),
    prev_det_col (P,3): previous-frame detection positions + colours; affine_by_id
    {track_id: (a, B, alpha)} or None. Padded slots use a huge offset so their
    predicted neighbour lands far from any detection (never an inlier); kcount
    holds the real neighbour count for the inlier-fraction denominator.
    Returns dict: off (M,kmax,2), col (M,kmax,3), B (M,2,2), alpha (M,), kcount (M,).
    """
    M = len(track_ids)
    off = np.full((M, kmax, 2), 1e7, dtype=np.float64)
    col = np.zeros((M, kmax, 3), dtype=np.float64)
    Bs = np.zeros((M, 2, 2), dtype=np.float64)
    al = np.zeros(M, dtype=np.float64)
    kc = np.zeros(M, dtype=np.int64)
    prev_det_xy = np.asarray(prev_det_xy, dtype=np.float64)
    prev_det_col = np.asarray(prev_det_col, dtype=np.float64)
    for i in range(M):
        tid = int(track_ids[i])
        nbrs = neighbor_lists.get(tid, None)
        if not nbrs:
            continue
        nbrs = np.asarray(nbrs, dtype=np.int64)
        if nbrs.size > kmax:
            d = np.linalg.norm(prev_det_xy[nbrs] - track_ref_xy[i], axis=1)
            nbrs = nbrs[np.argsort(d, kind="stable")[:kmax]]
        k = int(nbrs.size)
        kc[i] = k
        off[i, :k] = prev_det_xy[nbrs] - track_ref_xy[i]
        col[i, :k] = prev_det_col[nbrs]
        if affine_by_id is not None:
            aff = affine_by_id.get(tid)
            if aff is not None:
                _a, B, alpha = aff
                Bs[i] = np.asarray(B, dtype=np.float64)
                al[i] = float(alpha)
    return {"off": off, "col": col, "B": Bs, "alpha": al, "kcount": kc}


def graph_struct_cc_for_pairs(i_indices, j_indices, det_xy, det_colors, gs,
                              *, gate_px=8.0, colour_max=60.0, det_tree=None):
    """Structural appearance score per gated (track i, candidate j) pair, for the
    cost matrix. Returns (scores (B,), valid (B,) bool). valid = track has a
    star-graph (kcount>0); invalid pairs fall back to dist+color in the caller."""
    i_arr = np.asarray(i_indices, dtype=np.int64)
    j_arr = np.asarray(j_indices, dtype=np.int64)
    det_xy = np.asarray(det_xy, dtype=np.float64)
    anchors = det_xy[j_arr]
    scores = graph_struct_scores_batch(
        anchors, gs["off"][i_arr], gs["col"][i_arr], det_xy, det_colors,
        Bs=gs["B"][i_arr], alphas=gs["alpha"][i_arr], gate_px=gate_px,
        colour_max=colour_max, kcounts=gs["kcount"][i_arr], det_tree=det_tree)
    valid = gs["kcount"][i_arr] > 0
    import os as _os
    if (_os.environ.get("CBTV_GS_DEBUG") or _os.environ.get("CPTV_GS_DEBUG")) and scores.size:
        import sys as _sys
        _v = scores[valid] if valid.any() else scores
        # per-track score spread: if ~0 everywhere -> gate too tight (n_matched~0);
        # if high but uniform across a track's candidates -> can't discriminate.
        print(f"[GS_DEBUG] B={scores.size} valid={int(valid.sum())} "
              f"score mean={float(_v.mean()):.3f} std={float(_v.std()):.3f} "
              f"max={float(_v.max()):.3f} frac>0.1={float((_v>0.1).mean()):.2f} "
              f"meanK={float(gs['kcount'][i_arr].mean()):.1f}", file=_sys.stderr, flush=True)
    return scores, valid


def graph_struct_scores_batch(anchors, star_offsets, star_colors,
                              det_xy, det_colors, *, Bs=None, alphas=None,
                              gate_px=8.0, colour_max=60.0, kcounts=None, det_tree=None):
    """VECTORISED structural scores for MANY (track, candidate) pairs at once --
    the cost-matrix path (the per-pair graph_struct_score is too slow at ~2270
    pairs/frame: per-call scipy Hungarian + KDTree dominate). For each pair,
    anchor the star-graph at the candidate, deform by the local transform B, and
    GREEDY-match each predicted neighbour to its nearest detection within gate_px
    AND colour_max (the colour gate is the moat). Score = inlier_fraction x
    residual_factor (a vectorised slice of cluster_confidence -- drops the margin
    term for speed; the colour gate still rejects lattice shifts).

    anchors (P,2); star_offsets (P,k,2) neighbour offsets rel to track in frame n;
    star_colors (P,k,3); det_xy (N,2); det_colors (N,3); Bs (P,2,2) or None;
    alphas (P,) or None. Returns scores (P,) in [0,1]. Greedy (not 1:1 Hungarian)
    -- an approximation acceptable for a cost term, ~10x cheaper.
    """
    from scipy.spatial import cKDTree
    anchors = np.asarray(anchors, dtype=np.float64)
    off = np.asarray(star_offsets, dtype=np.float64)
    P, k = off.shape[0], off.shape[1]
    if P == 0:
        return np.zeros(0)
    if Bs is not None and alphas is not None:
        Boff = np.einsum('pij,pkj->pki', np.asarray(Bs, dtype=np.float64), off)
        deformed = off + np.asarray(alphas, dtype=np.float64)[:, None, None] * Boff
    else:
        deformed = off
    pred = anchors[:, None, :] + deformed                      # (P,k,2)
    tree = det_tree if det_tree is not None else cKDTree(np.asarray(det_xy, dtype=np.float64))
    dist, idx = tree.query(pred.reshape(-1, 2))                # nearest det per predicted neighbour
    dist = dist.reshape(P, k); idx = idx.reshape(P, k)
    matched_col = np.asarray(det_colors, dtype=np.float64)[idx]  # (P,k,3)
    cdist = np.sqrt(((matched_col - np.asarray(star_colors, dtype=np.float64)) ** 2).sum(axis=2))
    ok = (dist <= gate_px) & (cdist <= colour_max)              # (P,k)
    n_matched = ok.sum(axis=1).astype(np.float64)
    # denominator = real neighbour count per pair (kcounts) so huge-offset PADDED
    # slots (which never match) do not dilute the inlier fraction; default = k.
    denom = np.maximum(np.asarray(kcounts, dtype=np.float64), 1.0) if kcounts is not None else max(k, 1)
    inlier_frac = n_matched / denom
    msum = np.where(ok, dist, 0.0).sum(axis=1)
    mean_res = np.where(n_matched > 0, msum / np.maximum(n_matched, 1.0), gate_px)
    res_factor = np.exp(-mean_res / max(0.5 * gate_px, 1e-9))
    return inlier_frac * (0.3 + 0.7 * res_factor)
