"""Colour constellations — Phase 3 Stage C.

Match coherent colour clusters of HARD (not-yet-matched) tracks as single
affine units against the detection set, commit members above the residual
bar jointly, and feed confident cluster transforms back into the field.

Why this attacks SWAP: members of a committed cluster cannot exchange
targets — the one-to-one joint assignment inside the cluster forbids
intra-neighbourhood swaps by construction, and one measurement settles
5-12 particles (accuracy AND compute from the same operation).

Design (plan §6 Phase 3, design-chat §13/§16/§17):
  1. Coherence graph over active tracks: kNN edges weighted by distance,
     previous-velocity agreement, field smoothness across the edge, and
     nearest-candidate displacement compatibility. Weak edges mark shear
     boundaries — clusters never cross them.
  2. Seed from hard particles (highest local ambiguity first), grow along
     strong edges, keep clusters compact / non-collinear / 5-12 members,
     prefer growth that raises colour distinctiveness (mixed-colour
     constellations are the moat — a monochrome cluster is geometrically
     ambiguous in dense same-colour neighbourhoods, a mixed one is not).
  3. Transform hypotheses, cheapest first: field-predicted translation ->
     +small rotation -> +local affine refit from provisional inliers.
     Scored at DETECTION CENTRES (distance residual + centre-colour
     compatibility + one-to-one target usage + outlier count). No raster
     correlation (explicitly deferred to Phase 4).
  4. Graded output, never all-or-nothing: per-member residuals; commit
     only members above the residual bar; outliers fall through to the
     normal cost-matrix path. Clusters partially break under shear —
     expected, never forced.
  5. Cluster->field feedback: committed member displacements re-enter the
     local-affine fit of nearby non-members as high-weight anchors, and
     their distance gates re-centre on the updated predictions.

All heavy lifting is in pure functions; ``run_constellations`` is the
single impure orchestrator the runner calls (one line in run_tracker).
No module-level globals; CPU-only numpy/scipy (the per-frame workload is
a few hundred tracks — no GPU needed).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .field import fit_local_affine

_BIG = 1e9


# ---------------------------------------------------------------------------
# 1. Coherence graph
# ---------------------------------------------------------------------------

def build_coherence_graph(
    positions: np.ndarray,
    radii: np.ndarray,
    velocities: np.ndarray,
    affines: list,
    cand_disp: np.ndarray,
    *,
    k_neighbors: int = 8,
    w_min: float = 0.25,
):
    """Build the within-frame coherence graph over tracks.

    Parameters
    ----------
    positions : (M, 2) float — track positions (previous-frame coords).
    radii : (M,) float.
    velocities : (M, 2) float — previous-frame velocity per track; rows of
        NaN mean "unknown".
    affines : list of M entries, each ``(a, B, alpha)`` or None — the
        Phase-2 field per track (None / alpha=0 -> no field info).
    cand_disp : (M, 2) float — displacement from the track's predicted
        position to its nearest free candidate detection; NaN = none.
    k_neighbors : edges per node before weight pruning.
    w_min : edges below this weight are dropped (shear boundaries).

    Returns
    -------
    adj : list of M lists of (j, w) — symmetric adjacency.
    """
    M = positions.shape[0]
    adj: list = [[] for _ in range(M)]
    if M < 2:
        return adj
    from scipy.spatial import cKDTree

    mean_r = float(np.mean(radii)) if M else 8.0
    sigma_d = max(4.0 * mean_r, 8.0)
    sigma_v = max(mean_r, 2.0)
    sigma_f = max(0.5 * mean_r, 2.0)
    sigma_c = max(mean_r, 2.0)

    kdt = cKDTree(positions)
    k = min(k_neighbors + 1, M)
    dists, idxs = kdt.query(positions, k=k)

    # [Phase 4 Stage 1] Vectorized edge scoring (was a per-edge python loop
    # with math.exp — ~5k iterations/frame). Outcomes parity-checked against
    # the loop version on the GPU smoke (identical commits).
    src = np.repeat(np.arange(M), k - 1)
    dst = idxs[:, 1:].ravel().astype(np.int64)
    dd = dists[:, 1:].ravel()
    ok = (dst != src) & (dst < M)
    src, dst, dd = src[ok], dst[ok], dd[ok]
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    _, first = np.unique(a * np.int64(M) + b, return_index=True)
    ii, jj, dd = a[first], b[first], dd[first]

    w = np.exp(-(dd / sigma_d) ** 2)

    vi, vj = velocities[ii], velocities[jj]
    both_v = np.isfinite(vi).all(axis=1) & np.isfinite(vj).all(axis=1)
    dv = np.nan_to_num(np.hypot(vi[:, 0] - vj[:, 0], vi[:, 1] - vj[:, 1]))
    w = w * np.where(both_v, np.exp(-(dv / sigma_v) ** 2), 0.8)

    has_f = np.array([aff is not None and aff[2] > 0.0 for aff in affines], dtype=bool)
    if has_f.any():
        A = np.zeros((M, 2)); B = np.zeros((M, 2, 2)); AL = np.zeros(M)
        for t_idx in np.where(has_f)[0]:
            aff = affines[t_idx]
            A[t_idx] = aff[0]; B[t_idx] = aff[1]; AL[t_idx] = aff[2]
        both_f = has_f[ii] & has_f[jj]
        mid = 0.5 * (positions[ii] + positions[jj])
        ui = AL[ii, None] * (A[ii] + np.einsum('nij,nj->ni', B[ii], mid - positions[ii]))
        uj = AL[jj, None] * (A[jj] + np.einsum('nij,nj->ni', B[jj], mid - positions[jj]))
        df = np.hypot(ui[:, 0] - uj[:, 0], ui[:, 1] - uj[:, 1])
        w = w * np.where(both_f, np.exp(-(df / sigma_f) ** 2), 1.0)

    ci, cj = cand_disp[ii], cand_disp[jj]
    both_c = np.isfinite(ci).all(axis=1) & np.isfinite(cj).all(axis=1)
    dc = np.nan_to_num(np.hypot(ci[:, 0] - cj[:, 0], ci[:, 1] - cj[:, 1]))
    w = w * np.where(both_c, np.exp(-(dc / sigma_c) ** 2), 1.0)

    keep = w >= w_min
    for i_, j_, w_ in zip(ii[keep].tolist(), jj[keep].tolist(), w[keep].tolist(), strict=False):
        adj[i_].append((j_, w_))
        adj[j_].append((i_, w_))
    return adj


# ---------------------------------------------------------------------------
# 2. Seed and grow
# ---------------------------------------------------------------------------

def colour_novelty(colors: np.ndarray, members: list, j: int) -> float:
    """How much colour variety member j would add: min BGR distance to the
    current members' colours, normalised to [0, 1] by 255."""
    if not members:
        return 0.0
    dmin = min(float(np.linalg.norm(colors[j] - colors[m])) for m in members)
    return min(dmin / 255.0, 1.0)


def cluster_spread_ok(positions: np.ndarray, members: list, mean_r: float) -> bool:
    """Reject essentially collinear clusters (cannot pin rotation/shear)."""
    if len(members) < 3:
        return False
    pts = positions[members]
    c = pts.mean(axis=0)
    s = np.linalg.svd(pts - c, compute_uv=False)
    return float(s[-1]) >= 0.2 * mean_r


def grow_cluster(
    seed: int,
    adj: list,
    positions: np.ndarray,
    colors: np.ndarray,
    hard_set: set,
    *,
    mean_r: float,
    min_size: int = 5,
    max_size: int = 12,
    diam_cap_factor: float = 12.0,
    w_grow_min: float = 0.3,
    colour_bonus: float = 0.15,
    pdist: np.ndarray | None = None,
    cdist_colour: np.ndarray | None = None,
):
    """Grow a cluster from `seed` along strong coherence edges.

    Greedy frontier growth: at each step add the hard track with the best
    (edge strength to cluster) + colour-novelty bonus, subject to the
    diameter cap and the weak-edge (shear-boundary) bar. Returns the
    member list (>= min_size) or None.

    pdist: optional (M, M) pairwise-distance cache ([Phase 4 Stage 1] —
    the per-candidate hypot loop dominated growth cost; the cache turns
    the diameter check into an O(size) lookup). None = compute on the fly
    (identical results).
    """
    diam_cap = diam_cap_factor * mean_r
    members = [seed]
    member_set = {seed}
    while len(members) < max_size:
        # [Phase 5 Stage 1] Gather the scan-ordered frontier edges (member
        # append order, then adj stored order) with the static per-edge
        # filters inline — this is a cheap python pass; the EXPENSIVE part
        # (the per-edge pdist.max / cdist.min numpy calls — ~the whole
        # {reduce} cost) is then batched into two vectorized reductions.
        cjl, cwl = [], []
        for m in members:
            for (j, w) in adj[m]:
                if j in member_set or j not in hard_set:
                    continue
                if w < w_grow_min:
                    continue
                cjl.append(j)
                cwl.append(w)
        if not cjl:
            break
        cj = np.asarray(cjl, dtype=np.int64)
        cw = np.asarray(cwl, dtype=np.float64)
        mem_arr = np.asarray(members, dtype=np.int64)
        # Diameter cap (compactness — local-affine validity), batched.
        if pdist is not None:
            dmax = pdist[cj][:, mem_arr].max(axis=1)
        else:
            dd = positions[cj][:, None, :] - positions[mem_arr][None, :, :]
            dmax = np.hypot(dd[..., 0], dd[..., 1]).max(axis=1)
        # Colour-novelty bonus, batched. float64 (matches the original
        # float() cast) so scores are bit-identical -> argmax picks the same
        # winner (np.argmax returns the first max == the strict-> first-wins).
        if cdist_colour is not None:
            nov = np.minimum(
                cdist_colour[cj][:, mem_arr].min(axis=1).astype(np.float64) / 255.0,
                1.0)
        else:
            nov = np.array([colour_novelty(colors, members, int(j)) for j in cjl],
                           dtype=np.float64)
        score = cw + colour_bonus * nov
        score = np.where(dmax <= diam_cap, score, -np.inf)   # diam-fail can't win
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break
        bj = int(cj[best])
        members.append(bj)
        member_set.add(bj)
    if len(members) < min_size:
        return None
    if not cluster_spread_ok(positions, members, mean_r):
        return None
    return members


# ---------------------------------------------------------------------------
# 3. Transform hypotheses + detection-centre scoring
# ---------------------------------------------------------------------------

def _rotate_about(pts: np.ndarray, centre: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return centre + (pts - centre) @ R.T


def score_hypothesis(
    pred_xy: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    det_xy: np.ndarray,
    det_colors: np.ndarray,
    free_dets: np.ndarray,
    *,
    gate_px: float,
    colour_max: float,
    lambda_colour: float = 0.5,
    cd2: np.ndarray | None = None,
):
    """Score one transform hypothesis at detection centres.

    For each member (predicted position pred_xy[i]), compatible detections
    are free dets within gate_px AND with BGR distance <= colour_max.
    One-to-one inside the cluster via Hungarian on
    cost = residual_px + lambda_colour * gate_px * (colour_dist/colour_max).

    Returns dict with: assign (k,) det index or -1; residual (k,) px;
    margin (k,) second-best-minus-chosen cost (inf if unique); n_matched.
    """
    from scipy.optimize import linear_sum_assignment

    k = pred_xy.shape[0]
    n = det_xy.shape[0]
    assign = np.full(k, -1, dtype=np.int64)
    residual = np.full(k, np.inf, dtype=np.float64)
    margin = np.full(k, np.inf, dtype=np.float64)
    if n == 0:
        return {"assign": assign, "residual": residual, "margin": margin, "n_matched": 0}

    # [Phase 5 Stage 1] Vectorized over members (was a per-member python loop
    # — ~45% of t_const). Every per-element op below is identical to the loop
    # version, so the values are bit-identical; the ONLY parity-sensitive
    # choice is the column order fed to the Hungarian (tie-breaking can differ
    # under column permutation), so the union is rebuilt in the EXACT
    # first-appearance scan order the loop produced (sort used columns by the
    # first member row that sees them, then by column index).
    # [Phase 4 precision] Gate in SQUARED space (bit-exact decisions) and take
    # sqrt only on the survivors.
    gate2 = gate_px * gate_px
    cmax2 = colour_max * colour_max
    dx = det_xy[:, 0][None, :] - pred_xy[:, 0][:, None]
    dy = det_xy[:, 1][None, :] - pred_xy[:, 1][:, None]
    D2 = dx * dx + dy * dy                                   # (k, n)
    if cd2 is None:
        diffc = det_colors[None, :, :] - colors[:, None, :]
        CD2 = np.einsum('knd,knd->kn', diffc, diffc)        # (k, n)
    else:
        CD2 = cd2                                           # shared across hyps
    okmat = free_dets[None, :] & (D2 <= gate2) & (CD2 <= cmax2)
    used = okmat.any(axis=0)
    if not used.any():
        return {"assign": assign, "residual": residual, "margin": margin, "n_matched": 0}

    first_i = np.argmax(okmat, axis=0)                       # first True row per col
    union_cols = np.where(used)[0]
    union_cols = union_cols[np.lexsort((union_cols, first_i[union_cols]))]

    oksub = okmat[:, union_cols]
    Dsub = np.sqrt(D2[:, union_cols])
    Csub = Dsub + lambda_colour * gate_px * (np.sqrt(CD2[:, union_cols]) / max(colour_max, 1e-9))
    C = np.where(oksub, Csub, _BIG)
    R = np.where(oksub, Dsub, np.inf)

    rows, cols = linear_sum_assignment(C)
    sel = C[rows, cols] < _BIG
    rr, cc = rows[sel], cols[sel]
    assign[rr] = union_cols[cc]
    residual[rr] = R[rr, cc]
    # Margin = (2nd-best - best) finite cost per matched row (unconstrained).
    # With a single union column every matched row has one finite cost ->
    # margin stays inf (the init), matching the loop's better.size<=1 branch.
    # Sort only the matched rows (unmatched rows would give inf-inf=nan).
    if rr.size and C.shape[1] >= 2:
        Csort = np.sort(np.where(C[rr] < _BIG, C[rr], np.inf), axis=1)
        margin[rr] = Csort[:, 1] - Csort[:, 0]              # inf if <2 finite
    n_matched = int((assign >= 0).sum())
    return {"assign": assign, "residual": residual, "margin": margin, "n_matched": n_matched}


def cluster_confidence(
    k: int,
    n_inliers: int,
    mean_inlier_residual: float,
    mean_margin: float,
    *,
    gate_px: float,
    prior_dev_px: float = 0.0,
    prior_weight: float = 0.0,
    sigma_dev_px: float = 8.0,
):
    """Graded cluster confidence in [0, 1].

    Four factors:
      * inlier fraction — partial breakage (shear) lowers it smoothly;
      * residual factor — how tightly the transform explains the inliers;
      * ambiguity factor — mean per-member assignment margin. This is the
        colour moat made operational: in a dense same-colour neighbourhood
        the second-best detection is nearly as good (margin ~ spacing
        only), while a mixed-colour constellation prices wrong-colour
        alternatives out via the colour term, inflating the margin.
        A geometrically perfect but ambiguous (monochrome, symmetric)
        cluster cannot exceed 0.7.
      * prior-deviation factor — the lattice-shift guard. A coherent
        WRONG assignment (whole cluster matched one site over in a dense
        quasi-regular region) has tiny residuals and unit margins (each
        member sees exactly one compatible det near its shifted spot);
        its signature is a SYSTEMATIC displacement between the winning
        assignment and the field-seeded prediction. prior_dev_px is that
        mean systematic shift; prior_weight (mean member field alpha)
        controls how much it counts: field confident -> a one-spacing
        deviation collapses confidence; field off/weak -> no penalty
        (colour distinctiveness must carry the match alone). Measured
        failure this guards (smoke, PRIMARY step 4): clusters committing
        5 members shifted onto neighbours at conf 0.61-0.63, residuals
        0.1-1.2 px, true particles 13-21 px away.
    """
    if k <= 0 or n_inliers <= 0:
        return 0.0
    inlier_frac = n_inliers / float(k)
    res_factor = math.exp(-max(mean_inlier_residual, 0.0) / max(0.5 * gate_px, 1e-9))
    m = 0.0 if not np.isfinite(mean_margin) else min(mean_margin / max(gate_px, 1e-9), 1.0)
    if not np.isfinite(mean_margin):
        m = 1.0
    amb_factor = 0.4 + 0.6 * m
    w = min(max(prior_weight, 0.0), 1.0)
    dev_factor = (1.0 - w) + w * math.exp(-(prior_dev_px / max(sigma_dev_px, 1e-9)) ** 2)
    return float(inlier_frac * (0.3 + 0.7 * res_factor) * amb_factor * dev_factor)


def match_cluster(
    member_xy: np.ndarray,
    member_r: np.ndarray,
    member_colors: np.ndarray,
    seed_pred_xy: np.ndarray,
    det_xy: np.ndarray,
    det_colors: np.ndarray,
    free_dets: np.ndarray,
    *,
    gate_px: float,
    colour_max: float,
    residual_bar_px: np.ndarray,
    rot_thetas: tuple = (-0.10, -0.05, 0.05, 0.10),
    prior_weight: float = 0.0,
    lambda_colour: float = 0.5,
):
    """Full graded cluster match: hypothesis tiers + detection-centre scoring.

    Tiers (cheapest first): seed prediction (field translation/deform) ->
    small rotations about the predicted centroid -> local affine refit
    from the best tier's provisional inliers (one iteration).

    Returns dict:
      assign (k,), residual (k,), inlier (k,) bool, conf float,
      transform (a, B) or None  — the recovered cluster affine
      (source-frame coords -> displacement), n_outliers.
    """
    k = member_xy.shape[0]

    # [Phase 4 Stage 1] Detection subsetting: every hypothesis (seed,
    # rotations <=0.10 rad, refit) predicts within rad + slack of the
    # predicted centroid, so dets outside that disc are incompatible for
    # ALL hypotheses (gate check would reject them) — score against the
    # subset and translate indices back. Identical outcomes, ~6-10x less
    # per-hypothesis work at N~600 dets.
    _centre = seed_pred_xy.mean(axis=0)
    _rad = float(np.max(np.hypot(seed_pred_xy[:, 0] - _centre[0],
                                 seed_pred_xy[:, 1] - _centre[1]))) if k else 0.0
    _slack = gate_px + 0.25 * _rad + 4.0
    _sub = np.where(np.hypot(det_xy[:, 0] - _centre[0],
                             det_xy[:, 1] - _centre[1]) <= _rad + _slack)[0]
    det_xy_s = det_xy[_sub]
    det_colors_s = det_colors[_sub]
    free_s = free_dets[_sub]

    # [Phase 5 Stage 1] Colour distance is hypothesis-independent (rotations
    # move predicted positions, not colours) — compute the (k x n_sub) colour
    # CD2 once and share it across all hypotheses. Bit-identical to the
    # per-hypothesis recompute.
    if det_xy_s.shape[0]:
        _diffc = det_colors_s[None, :, :] - member_colors[:, None, :]
        _cd2_shared = np.einsum('knd,knd->kn', _diffc, _diffc)
    else:
        _cd2_shared = None

    def _score(pred):
        return score_hypothesis(pred, member_r, member_colors, det_xy_s, det_colors_s,
                                free_s, gate_px=gate_px, colour_max=colour_max,
                                lambda_colour=lambda_colour, cd2=_cd2_shared)

    hyps = [("seed", seed_pred_xy, _score(seed_pred_xy))]
    centre = seed_pred_xy.mean(axis=0)
    for th in rot_thetas:
        p = _rotate_about(seed_pred_xy, centre, th)
        hyps.append((f"rot{th:+.2f}", p, _score(p)))

    def _rank(h):
        sc = h[2]
        inl = sc["residual"][sc["assign"] >= 0]
        mean_res = float(np.mean(inl)) if inl.size else np.inf
        return (-sc["n_matched"], mean_res)

    hyps.sort(key=_rank)
    best_name, best_pred, best_sc = hyps[0]

    # Affine refit from provisional matched pairs (>= 3), one iteration.
    transform = None
    matched = np.where(best_sc["assign"] >= 0)[0]
    if matched.size >= 3:
        src = member_xy[matched]
        dst = det_xy_s[best_sc["assign"][matched]]
        a, B, _res, n_used = fit_local_affine(
            member_xy.mean(axis=0), src, dst - src)
        if n_used >= 3:
            rel = member_xy - member_xy.mean(axis=0)
            pred_refit = member_xy + a[None, :] + rel @ B.T
            sc_refit = _score(pred_refit)
            if _rank(("refit", pred_refit, sc_refit)) <= _rank((best_name, best_pred, best_sc)):
                best_name, best_pred, best_sc = "refit", pred_refit, sc_refit
                transform = (a, B)

    assign = best_sc["assign"]
    residual = best_sc["residual"]
    inlier = (assign >= 0) & (residual <= residual_bar_px)
    inl_res = residual[inlier]
    inl_marg = best_sc["margin"][inlier]
    finite_marg = inl_marg[np.isfinite(inl_marg)]
    mean_margin = float(np.mean(finite_marg)) if finite_marg.size else np.inf
    # Lattice-shift guard: systematic shift between the winning assignment
    # and the field-seeded prediction (mean over inliers; rotation about
    # the centroid cancels in the mean, translation shifts do not).
    if inlier.any():
        _dev = det_xy_s[assign[inlier]] - seed_pred_xy[inlier]
        prior_dev = float(np.hypot(*np.mean(_dev, axis=0)))
    else:
        prior_dev = 0.0
    conf = cluster_confidence(
        k, int(inlier.sum()),
        float(np.mean(inl_res)) if inl_res.size else np.inf,
        mean_margin, gate_px=gate_px,
        prior_dev_px=prior_dev, prior_weight=prior_weight,
        sigma_dev_px=max(float(np.mean(member_r)), 4.0))

    # Recover the committed transform even if refit didn't run/win, from
    # the final inliers (for the field feedback + diagnostics).
    if transform is None and inlier.sum() >= 3:
        src = member_xy[inlier]
        dst = det_xy_s[assign[inlier]]
        a, B, _res, n_used = fit_local_affine(member_xy.mean(axis=0), src, dst - src)
        if n_used >= 3:
            transform = (a, B)

    # Translate subset-space det indices back to the caller's full space.
    assign_full = np.full(k, -1, dtype=np.int64)
    _m = assign >= 0
    if _m.any():
        assign_full[_m] = _sub[assign[_m]]

    return {
        "assign": assign_full, "residual": residual, "inlier": inlier,
        "conf": conf, "transform": transform,
        "n_outliers": int(k - inlier.sum()), "hypothesis": best_name,
    }


# ---------------------------------------------------------------------------
# 4. Cluster -> field feedback (+ nearby non-member gate re-centring)
# ---------------------------------------------------------------------------

def refresh_field_for_neighbours(
    tracks: list,
    matched_tracks: set,
    cluster_anchors: list,
    affine_by_id: dict,
    predicted_positions: np.ndarray,
    *,
    radius_px: float,
    anchor_weight: float = 3.0,
):
    """Re-fit the local affine of unmatched tracks near committed clusters.

    cluster_anchors: list of (x, y, vx, vy) from committed members (current
    positions + observed displacement). For each still-unmatched track
    within radius_px of any cluster anchor, re-fit its local affine over
    the velocity-anchor pool (all tracks with known velocity; matched =
    fresh observation, weight 2; unmatched = stale prior, weight 1) plus
    the cluster anchors at `anchor_weight`, then re-centre its predicted
    position (the [COST-1] gate centre). Mutates affine_by_id and
    predicted_positions in place. Returns #tracks refreshed.
    """
    if not cluster_anchors or predicted_positions is None or affine_by_id is None:
        return 0
    from scipy.spatial import cKDTree

    from .field import compute_anchor_spread, deformation_confidence, fit_local_affine, predict_position

    ca_xy = np.array([(a[0], a[1]) for a in cluster_anchors], dtype=np.float64)
    ca_kdt = cKDTree(ca_xy)

    pool_xy: Any        # built as lists, then converted to float64 arrays below (T041: mypy)
    pool_uv: Any
    pool_w: Any
    pool_xy, pool_uv, pool_w = [], [], []
    for j, t in enumerate(tracks):
        if t.px is None or t.py is None:
            continue
        pool_xy.append((t.x, t.y))
        pool_uv.append((t.x - t.px, t.y - t.py))
        pool_w.append(2.0 if j in matched_tracks else 1.0)
    for (x, y, vx, vy) in cluster_anchors:
        pool_xy.append((x, y)); pool_uv.append((vx, vy)); pool_w.append(anchor_weight)
    if len(pool_xy) < 3:
        return 0
    pool_xy = np.asarray(pool_xy, dtype=np.float64)
    pool_uv = np.asarray(pool_uv, dtype=np.float64)
    pool_w = np.asarray(pool_w, dtype=np.float64)
    pool_kdt = cKDTree(pool_xy)

    n_refreshed = 0
    for i, t in enumerate(tracks):
        if i in matched_tracks:
            continue
        if not ca_kdt.query_ball_point([t.x, t.y], r=radius_px):
            continue
        idxs = pool_kdt.query_ball_point([t.x, t.y], r=radius_px)
        if len(idxs) < 3:
            continue
        xy = pool_xy[idxs]; uv = pool_uv[idxs]; w = pool_w[idxs]
        a, B, res, n_used = fit_local_affine((t.x, t.y), xy, uv, weights=w)
        alpha = deformation_confidence(n_used, compute_anchor_spread(xy), res)
        if alpha <= 0.0:
            continue
        affine_by_id[t.id] = (a, B, float(alpha))
        xp = predict_position(a, B, (t.x, t.y), (t.x, t.y), float(alpha))
        predicted_positions[i, 0] = float(xp[0])
        predicted_positions[i, 1] = float(xp[1])
        n_refreshed += 1
    return n_refreshed


# ---------------------------------------------------------------------------
# 5. Orchestrator (the one impure entry point the runner calls)
# ---------------------------------------------------------------------------

def run_constellations(
    tracks: list,
    matched_tracks: set,
    matched_dets: set,
    all_candidates: list,
    colors_all,
    affine_by_id: dict | None,
    predicted_positions: np.ndarray | None,
    args,
    *,
    frame_idx: int = 0,
    log_writer=None,
    max_clusters_per_frame: int = 64,
    excluded=None,
):
    """Propose, score, and commit colour constellations for one frame.

    Mutates tracks / matched_tracks / matched_dets exactly like the
    MAIN-9 assignment block (incl. the --colour-ema rule), updates
    predicted_positions + affine_by_id in place via the cluster->field
    feedback, and returns a stats dict for profiling.

    [Phase 4 Stage 2a] Coverage levers, all parity-default:
      --cluster-max-per-frame (default 64) proposal cap;
      --cluster-rounds (default 1) multi-round passes — after a round's
        commits refresh the field, re-propose on the residual hard set
        (the round-2 seeds benefit from the round-1 anchors);
      --cluster-fallback-min-size (default 0=off; 3/4) one extra pass of
        small clusters at the stricter --cluster-fallback-conf bar for
        block particles no 5-cluster reaches (weaker joint constraint =
        lattice-shift risk back on the table: min-prior gate + deviation
        factor stay armed; the commit audit decides graduation).
    """
    stats = {"proposed": 0, "accepted": 0, "members_committed": 0,
             "outliers": 0, "field_refreshed": 0, "committed_track_ids": []}
    M = len(tracks)
    N = len(all_candidates)
    if M == 0 or N == 0:
        return stats

    min_size = int(getattr(args, "cluster_min_size", 5))
    max_size = int(getattr(args, "cluster_max_size", 12))
    commit_conf = float(getattr(args, "cluster_commit_conf", 0.6))
    cap = int(getattr(args, "cluster_max_per_frame", max_clusters_per_frame))
    rounds = max(1, int(getattr(args, "cluster_rounds", 1)))
    fb_min = int(getattr(args, "cluster_fallback_min_size", 0))
    fb_conf = float(getattr(args, "cluster_fallback_conf", 0.75))
    min_prior = 0.2

    # [Phase 4 Stage 2d] Displacement-aware commit bar: at observed
    # displacement >= ~0.8 radii/frame the lattice-shift risk grows
    # (Phase-3 audit: 98-99% at steps 3-4 but 87-88% at step 8) — raise
    # the commit-conf bar and the min-prior gate from the OBSERVED ratio
    # (never the benchmark step). Schedule: <0.8 base; 0.8-1.4 +0.05 conf
    # / +0.10 prior; >1.4 +0.10 / +0.20.
    if getattr(args, "cluster_disp_aware_bar", 0) == 1:
        from .field import observed_displacement_ratio
        _ratio = observed_displacement_ratio([], tracks)
        if _ratio > 1.4:
            commit_conf += 0.10; fb_conf += 0.10; min_prior += 0.20
        elif _ratio >= 0.8:
            commit_conf += 0.05; fb_conf += 0.05; min_prior += 0.10

    cid = 0
    for _rnd in range(rounds):
        d, cid = _constellation_pass(
            tracks, matched_tracks, matched_dets, all_candidates, colors_all,
            affine_by_id, predicted_positions, args,
            frame_idx=frame_idx, log_writer=log_writer, cap=cap,
            min_size=min_size, max_size=max_size, commit_conf=commit_conf,
            min_prior=min_prior, cluster_id_start=cid, excluded=excluded)
        for k_ in stats:
            stats[k_] += d[k_]
        if d["members_committed"] == 0:
            break
    if fb_min >= 3:
        d, cid = _constellation_pass(
            tracks, matched_tracks, matched_dets, all_candidates, colors_all,
            affine_by_id, predicted_positions, args,
            frame_idx=frame_idx, log_writer=log_writer, cap=cap,
            min_size=fb_min, max_size=max_size, commit_conf=fb_conf,
            min_prior=min_prior, cluster_id_start=cid, excluded=excluded)
        for k_ in stats:
            stats[k_] += d[k_]
    return stats


def _constellation_pass(
    tracks, matched_tracks, matched_dets, all_candidates, colors_all,
    affine_by_id, predicted_positions, args, *,
    frame_idx, log_writer, cap, min_size, max_size, commit_conf,
    cluster_id_start, min_prior=0.2, excluded=None,
):
    """One propose->score->commit->feedback pass over the current residual
    hard set. Snapshots track state fresh on entry, so successive rounds
    see post-commit positions/velocities and the refreshed field."""
    stats = {"proposed": 0, "accepted": 0, "members_committed": 0,
             "outliers": 0, "field_refreshed": 0, "committed_track_ids": []}
    M = len(tracks)
    N = len(all_candidates)
    residual_bar_frac = float(getattr(args, "cluster_residual_bar", 0.6))
    colour_max = float(getattr(args, "max_color", 60.0))
    max_dist = float(getattr(args, "max_dist", 20.0))
    # [Phase 4 Stage 3] Colour-blind science cell: no colour gate, no colour
    # cost, no colour-novelty growth — measures the moat on real video.
    _colour_blind = getattr(args, "cluster_colour_blind", 0) == 1
    if _colour_blind:
        colour_max = 1e9

    positions = np.array([(t.x, t.y) for t in tracks], dtype=np.float64)
    radii = np.array([t.r for t in tracks], dtype=np.float64)
    colors_trk = np.array([t.color_bgr for t in tracks], dtype=np.float64)
    mean_r = float(np.mean(radii)) if M else 8.0
    gate_px = max(0.75 * max_dist, 2.0 * mean_r)

    velocities = np.full((M, 2), np.nan, dtype=np.float64)
    for i, t in enumerate(tracks):
        if t.px is not None and t.py is not None:
            velocities[i] = (t.x - t.px, t.y - t.py)

    affines = [None] * M
    if affine_by_id:
        for i, t in enumerate(tracks):
            affines[i] = affine_by_id.get(t.id)

    det_xy = np.array([(d[0], d[1]) for d in all_candidates], dtype=np.float64)
    det_colors = np.array([colors_all[j] for j in range(N)], dtype=np.float64)
    free_dets = np.ones(N, dtype=bool)
    for j in matched_dets:
        if 0 <= j < N:
            free_dets[j] = False

    hard = [i for i in range(M) if i not in matched_tracks]
    if excluded:
        # [In/outflow C/D] exited / occluded tracks never join a constellation
        hard = [i for i in hard if i not in excluded]
    if len(hard) < min_size:
        return stats, cluster_id_start
    hard_set = set(hard)

    # Per-track seed predictions + per-track prior strength.
    # Prior strength s_i feeds the lattice-shift guard: field-fitted tracks
    # get their alpha; unfitted tracks fall back to their own previous
    # velocity as the seed (strength 0.5); tracks with neither (fresh
    # spawns, frame 1) get strength 0 — and a cluster without prior
    # coverage must NOT commit (see min-prior gate below): a missed commit
    # just falls through to the cost matrix, a wrong commit manufactures
    # a swap. This is the measured failure mode: wrong clusters sat
    # exactly on alpha=0 tracks whose zero-motion seed a shifted match
    # fits perfectly.
    prior_strength = np.zeros(M, dtype=np.float64)
    vfin = np.isfinite(velocities[:, 0])
    if predicted_positions is not None:
        pred_all = np.asarray(predicted_positions, dtype=np.float64).copy()
        for i in range(M):
            a_i = affines[i]
            if a_i is not None and a_i[2] > 0.0:
                prior_strength[i] = float(a_i[2])
            elif vfin[i]:
                # Field never engaged for this track — use its own velocity
                # as the seed (cluster-local only; the cost-matrix gate
                # keeps the original predicted_positions).
                pred_all[i] = positions[i] + velocities[i]
                prior_strength[i] = 0.5
    else:
        pred_all = positions.copy()
        pred_all[vfin] += velocities[vfin]
        prior_strength[vfin] = 0.5

    # Nearest free-candidate displacement (for graph compatibility + seed
    # ambiguity ordering). [Phase 4 Stage 1] vectorized: return_length
    # counts + one batched nearest query (was a per-hard-track loop).
    from scipy.spatial import cKDTree
    free_idx = np.where(free_dets)[0]
    cand_disp = np.full((M, 2), np.nan, dtype=np.float64)
    ambiguity = np.zeros(M, dtype=np.int64)
    if free_idx.size:
        free_kdt = cKDTree(det_xy[free_idx])
        hard_arr = np.asarray(hard, dtype=np.int64)
        counts = free_kdt.query_ball_point(pred_all[hard_arr], r=max_dist,
                                           return_length=True)
        ambiguity[hard_arr] = counts
        with_cand = hard_arr[np.asarray(counts) > 0]
        if with_cand.size:
            _dd, jj = free_kdt.query(pred_all[with_cand])
            cand_disp[with_cand] = det_xy[free_idx[jj]] - positions[with_cand]

    adj = build_coherence_graph(positions, radii, velocities, affines, cand_disp)

    # [Phase 4 Stage 1] Pairwise caches for growth (M~600 -> ~3 MB each,
    # recomputed per frame): spatial distances (diameter checks) and
    # colour distances (novelty was 60% of cluster runtime as 400k+
    # per-pair norm calls).
    # [Phase 4 precision] float32 caches — colours are uint8-origin (EMA
    # makes them fractional, but float32's ~1e-5 relative error is 4+
    # orders below any gate sensitivity) and the diameter checks compare
    # against px-scale thresholds; float64 was pure memory/bandwidth waste.
    from scipy.spatial.distance import cdist
    _pdist = cdist(positions.astype(np.float32), positions.astype(np.float32))
    _cdist_col = cdist(colors_trk.astype(np.float32), colors_trk.astype(np.float32))

    # Seeds: hard tracks, most candidate-ambiguous first (they need the
    # joint constraint most; trivial particles don't need a cluster).
    seeds = sorted(hard, key=lambda i: -ambiguity[i])

    proposals = []
    absorbed: set = set()
    for s in seeds:
        if len(proposals) >= cap:
            break
        if s in absorbed or ambiguity[s] == 0:
            continue
        members = grow_cluster(s, adj, positions, colors_trk, hard_set,
                               mean_r=mean_r, min_size=min_size, max_size=max_size,
                               pdist=_pdist, cdist_colour=_cdist_col,
                               colour_bonus=0.0 if _colour_blind else 0.15)
        if members is None:
            continue
        proposals.append(members)
        absorbed.update(members)
    stats["proposed"] = len(proposals)
    if not proposals:
        return stats, cluster_id_start

    # Score all proposals. Clusters with no prior coverage are dropped
    # outright (min-prior commit gate): without field or velocity priors
    # the lattice-shift guard is blind and a coherent wrong assignment is
    # indistinguishable from a right one in same-colour regions.
    # min_prior is displacement-scaled under --cluster-disp-aware-bar.
    scored = []
    for members in proposals:
        m_arr = np.asarray(members)
        prior_w = float(np.mean(prior_strength[m_arr]))
        if prior_w < min_prior:
            continue
        res = match_cluster(
            positions[m_arr], radii[m_arr], colors_trk[m_arr],
            pred_all[m_arr], det_xy, det_colors, free_dets,
            gate_px=gate_px, colour_max=colour_max,
            residual_bar_px=residual_bar_frac * radii[m_arr],
            prior_weight=prior_w,
            lambda_colour=0.0 if _colour_blind else 0.5,
        )
        scored.append((res["conf"], members, res))

    # Greedy acceptance by confidence with one-to-one target locking.
    scored.sort(key=lambda x: -x[0])
    beta = float(getattr(args, "colour_ema", 0.3))
    beta = min(max(beta, 0.0), 1.0)
    cluster_anchors = []
    cluster_id = cluster_id_start
    for conf, members, res in scored:
        if conf < commit_conf:
            if log_writer is not None:
                for mi, i in enumerate(members):
                    log_writer.writerow([frame_idx, f"rej{cluster_id}", tracks[i].id,
                                         "", "", f"{res['residual'][mi]:.2f}" if np.isfinite(res['residual'][mi]) else "",
                                         f"{conf:.3f}", 0])
            cluster_id += 1
            continue
        committed_any = False
        for mi, i in enumerate(members):
            if not res["inlier"][mi]:
                stats["outliers"] += 1
                continue
            j = int(res["assign"][mi])
            if i in matched_tracks or j in matched_dets or not free_dets[j]:
                continue  # superseded by a more confident cluster
            t = tracks[i]
            x, y, r, _s = all_candidates[j]
            vx, vy = float(x) - t.x, float(y) - t.y
            # Commit — mirrors the MAIN-9 update block (incl. colour EMA).
            t.px, t.py, t.pr = t.x, t.y, t.r
            t.x, t.y, t.r = float(x), float(y), 0.85 * t.r + 0.15 * float(r)
            old_c = np.asarray(t.color_bgr, dtype=np.float64)
            new_c = np.asarray(colors_all[j], dtype=np.float64)
            t.color_bgr = tuple((beta * new_c + (1.0 - beta) * old_c).tolist())
            t.missed = 0
            t.last_det_pos = (float(x), float(y))
            t.last_det_r = float(r)
            matched_tracks.add(i)
            matched_dets.add(j)
            free_dets[j] = False
            cluster_anchors.append((float(x), float(y), vx, vy))
            stats["members_committed"] += 1
            stats["committed_track_ids"].append(t.id)
            committed_any = True
            if log_writer is not None:
                log_writer.writerow([frame_idx, cluster_id, t.id,
                                     f"{x:.2f}", f"{y:.2f}",
                                     f"{res['residual'][mi]:.2f}", f"{conf:.3f}", 1])
        if committed_any:
            stats["accepted"] += 1
        cluster_id += 1

    # Cluster -> field feedback + nearby non-member gate re-centring.
    if cluster_anchors and predicted_positions is not None and affine_by_id is not None:
        stats["field_refreshed"] = refresh_field_for_neighbours(
            tracks, matched_tracks, cluster_anchors, affine_by_id,
            predicted_positions, radius_px=max(20.0, 6.0 * mean_r))
    return stats, cluster_id
