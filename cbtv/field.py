"""Local affine deformation field — Phase 2.

This module is intentionally a thin layer of *pure* numerical helpers.  All
state and orchestration lives in ``run_tracker.py``; the inner kernel and
cost-matrix builders accept the affine `(a, B, alpha)` per track as ordinary
kwargs (default = None / 0, so the call site is byte-identical when the
``--use-affine-field`` flag is off).

The local-affine model around track ``i`` says:

    u(x) ≈ a_i + B_i (x − x_i)            (displacement at position x)

where ``a_i`` is a translation and ``B_i`` is a 2×2 deformation matrix
(rotation + shear + scale).  We fit ``(a_i, B_i)`` from anchor pairs (a list
of (offset_from_centre, observed_displacement)) by weighted least squares.

The downstream consumers use two facts:

1. **Kernel build (kernels.py)** — when placing neighbour colour blobs at
   their offset ``r_n`` from the centre, replace ``r_n`` with the deformed
   offset ``r'_n = (I + alpha·B) r_n``.  ``alpha`` is a confidence in [0,1];
   ``alpha = 0`` reproduces the current behaviour exactly.
2. **Cost matrix (cost_matrix.py [COST-1/2])** — when computing the
   spatial distance and gate, re-centre the track on the predicted position
   ``x_pred = x_centre + alpha · a_i`` instead of the previous position,
   so the candidate set follows the predicted motion.

The chicken-and-egg ("field needs matches, matches need field") is broken
by sourcing anchors only from things that are known **before** the cost
matrix runs — previous-frame velocities (``Track.px/py``) and "easy"
current-frame matches committed *without* a kernel evaluation.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Core fit / apply / predict
# ---------------------------------------------------------------------------

def fit_local_affine(
    centre_xy: tuple,
    anchor_xy: np.ndarray,
    anchor_uv: np.ndarray,
    weights: np.ndarray | None = None,
):
    """Weighted least-squares fit of ``u(x) = a + B (x − centre)``.

    Solves the 6-unknown linear system

        u_x ≈ a_x + b11·dx + b12·dy
        u_y ≈ a_y + b21·dx + b22·dy

    for each anchor, with optional per-anchor weights.  Returns the fitted
    translation ``a``, deformation matrix ``B``, the weighted residual
    norm, and the number of anchors used.

    Parameters
    ----------
    centre_xy : (cx, cy) tuple of float
        The track's current centre — anchors are positioned relative to it.
    anchor_xy : (N, 2) float ndarray
        Absolute (x, y) positions of the N anchor sources in the *current*
        frame's coordinate system.
    anchor_uv : (N, 2) float ndarray
        Observed displacement (u, v) at each anchor (i.e. the anchor's own
        ``current_pos − previous_pos``).
    weights : (N,) float ndarray or None
        Per-anchor weights for the WLS solve.  ``None`` → uniform weights.

    Returns
    -------
    a : (2,) ndarray
        Fitted translation at the centre.
    B : (2, 2) ndarray
        Fitted deformation matrix (the linear part of the displacement field).
    residual : float
        Weighted RMS residual of the fit in pixels.
    n : int
        Number of anchors that contributed (after filtering NaNs/inf).
    """
    anchor_xy = np.asarray(anchor_xy, dtype=np.float64)
    anchor_uv = np.asarray(anchor_uv, dtype=np.float64)
    if anchor_xy.shape[0] == 0:
        return np.zeros(2, dtype=np.float64), np.zeros((2, 2), dtype=np.float64), 0.0, 0
    if anchor_xy.ndim != 2 or anchor_xy.shape[1] != 2 or anchor_uv.shape != anchor_xy.shape:
        raise ValueError("anchor_xy and anchor_uv must both be (N, 2)")

    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    dx = anchor_xy[:, 0] - cx
    dy = anchor_xy[:, 1] - cy
    ux = anchor_uv[:, 0]
    uy = anchor_uv[:, 1]

    finite = (
        np.isfinite(dx) & np.isfinite(dy) &
        np.isfinite(ux) & np.isfinite(uy)
    )
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        finite &= np.isfinite(weights) & (weights > 0)
    else:
        weights = np.ones_like(dx)

    if finite.sum() < 3:
        # Under-determined.  Fall back to pure translation = weighted mean velocity.
        mask = finite & (weights > 0)
        if mask.sum() == 0:
            return np.zeros(2, dtype=np.float64), np.zeros((2, 2), dtype=np.float64), 0.0, 0
        wsum = weights[mask].sum()
        a = np.array([
            (weights[mask] * ux[mask]).sum() / wsum,
            (weights[mask] * uy[mask]).sum() / wsum,
        ], dtype=np.float64)
        res = float(np.sqrt(
            (weights[mask] * ((ux[mask] - a[0]) ** 2 + (uy[mask] - a[1]) ** 2)).sum() / wsum
        ))
        return a, np.zeros((2, 2), dtype=np.float64), res, int(mask.sum())

    dx = dx[finite]; dy = dy[finite]
    ux = ux[finite]; uy = uy[finite]
    w = weights[finite]

    # Design matrix: rows are [1, dx, dy].  We solve two 3-unknown WLS
    # problems (one per output component) in a single block.
    A = np.stack([np.ones_like(dx), dx, dy], axis=1)
    Y = np.stack([ux, uy], axis=1)  # (n, 2)
    W = np.sqrt(w)[:, None]
    Aw = A * W
    Yw = Y * W
    # 3-unknown lstsq for each column of Y.  Same A, so a single solve works.
    sol, *_ = np.linalg.lstsq(Aw, Yw, rcond=None)  # (3, 2)
    a = sol[0, :].copy()              # (2,) = [a_x, a_y]
    B = np.array([
        [sol[1, 0], sol[2, 0]],       # row for u_x: [b11, b12]
        [sol[1, 1], sol[2, 1]],       # row for u_y: [b21, b22]
    ], dtype=np.float64)

    pred = A @ sol                    # (n, 2)
    err = Y - pred
    res_sq = (w * (err[:, 0] ** 2 + err[:, 1] ** 2)).sum() / max(w.sum(), 1e-12)
    residual = float(math.sqrt(max(res_sq, 0.0)))
    return a, B, residual, int(finite.sum())


def deformation_confidence(
    n: int,
    spread_px: float,
    residual_px: float,
    *,
    n_min: int = 4,
    n_sat: int = 12,
    spread_min: float = 3.0,
    spread_sat: float = 12.0,
    residual_max: float = 4.0,
) -> float:
    """Soft confidence in [0,1] that the fitted affine is trustworthy.

    Goes to 1 only when:
      * we used many anchors (``n >> n_min``),
      * the anchor cloud is spatially well-spread (not collinear),
      * the weighted residual is small.

    Goes to 0 when any of these is bad (under-determined, collinear,
    sparse, high residual).  The cutoffs are deliberate, conservative
    defaults — they're the only Phase-2 knobs.

    Parameters
    ----------
    n : int
        Number of anchors that contributed to the fit.
    spread_px : float
        Geometric spread of the anchor cloud — usually ``sqrt(var_x + var_y)``
        in pixels, or the half-extent of the convex hull, etc.  Smaller =
        more collinear / clustered.
    residual_px : float
        Weighted RMS residual reported by ``fit_local_affine``, in pixels.

    Other parameters control the soft thresholds; defaults are sensible
    for the cbtv corpus (particles ~ 6-10 px, ≥ 4 anchors needed to fit
    six params robustly).

    Returns
    -------
    alpha : float in [0, 1].
    """
    if n < 3:
        return 0.0

    def _smoothstep(x: float, lo: float, hi: float) -> float:
        if x <= lo: return 0.0
        if x >= hi: return 1.0
        t = (x - lo) / (hi - lo)
        return t * t * (3.0 - 2.0 * t)

    a_n = _smoothstep(float(n), float(n_min), float(n_sat))
    a_spread = _smoothstep(float(spread_px), float(spread_min), float(spread_sat))
    # Lower residual = higher confidence; smoothstep on -residual.
    if residual_px <= 0.0:
        a_res = 1.0
    else:
        # Convert "residual under residual_max" to a [0,1] confidence.
        a_res = max(0.0, 1.0 - min(1.0, residual_px / residual_max))
        # Soften (so residual_max/2 still gives ~0.5, not exactly 0.5)
        a_res = a_res * a_res * (3.0 - 2.0 * a_res)
    return float(max(0.0, min(1.0, a_n * a_spread * a_res)))


def apply_affine(
    B: np.ndarray,
    r_n: np.ndarray | tuple,
    alpha: float,
) -> np.ndarray:
    """Apply the deformation to a neighbour offset.

    ``r'_n = (I + alpha · B) r_n``.  When ``alpha == 0`` or ``B is None``
    this is the identity — the kernel placement reduces to the un-deformed
    case exactly.
    """
    if alpha == 0.0 or B is None:
        return np.asarray(r_n, dtype=np.float64).copy()
    B = np.asarray(B, dtype=np.float64)
    r = np.asarray(r_n, dtype=np.float64)
    return r + alpha * (B @ r)


def predict_position(
    a: np.ndarray | tuple,
    B: np.ndarray | None,
    centre_xy: np.ndarray | tuple,
    x: np.ndarray | tuple,
    alpha: float,
) -> np.ndarray:
    """Predicted *current-frame* position of a point at ``x``.

    Uses ``u(x) = alpha · (a + B (x - centre))`` so that with ``alpha = 0``
    the prediction collapses to the input ``x`` (no motion compensation).
    """
    x = np.asarray(x, dtype=np.float64)
    if alpha == 0.0:
        return x.copy()
    a = np.asarray(a, dtype=np.float64)
    centre = np.asarray(centre_xy, dtype=np.float64)
    if B is None:
        return x + alpha * a
    B = np.asarray(B, dtype=np.float64)
    return x + alpha * (a + B @ (x - centre))


# ---------------------------------------------------------------------------
# Anchor assembly — the chicken-and-egg break
# ---------------------------------------------------------------------------

def assemble_anchors(
    centre_xy: tuple,
    nbrs_prev_velocity: list,
    easy_matches: list,
    *,
    source: str = "prev_plus_easy",
    velocity_weight: float = 1.0,
    easy_weight: float = 2.0,
):
    """Build the anchor arrays from previous-frame velocity + easy matches.

    Parameters
    ----------
    centre_xy : (cx, cy)
        The current centre this affine is fitted around (track's own
        current position).
    nbrs_prev_velocity : list of (x_curr, y_curr, vx, vy)
        Other tracks that have a known previous velocity (i.e. their
        ``Track.px/py`` were set last frame).  Source = "prev".
    easy_matches : list of (x_curr, y_curr, vx, vy)
        Tracks just committed by the easy-match step (one colour-compatible
        candidate near the predicted position, no target competition).
        Source = "easy".
    source : {"prev_plus_easy", "prev", "easy"}
        Which anchor set to use.  Default ``prev_plus_easy``.
    velocity_weight, easy_weight : float
        Per-anchor weights.  Easy matches get higher weight by default
        because they're current-frame measurements, not just prior velocities.

    Returns
    -------
    anchor_xy : (N, 2) ndarray
    anchor_uv : (N, 2) ndarray
    weights   : (N,) ndarray
    """
    use_prev = source in ("prev_plus_easy", "prev")
    use_easy = source in ("prev_plus_easy", "easy")
    rows_xy, rows_uv, rows_w = [], [], []
    if use_prev:
        for (x, y, vx, vy) in nbrs_prev_velocity:
            rows_xy.append((x, y))
            rows_uv.append((vx, vy))
            rows_w.append(velocity_weight)
    if use_easy:
        for (x, y, vx, vy) in easy_matches:
            rows_xy.append((x, y))
            rows_uv.append((vx, vy))
            rows_w.append(easy_weight)
    if not rows_xy:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.asarray(rows_xy, dtype=np.float64),
        np.asarray(rows_uv, dtype=np.float64),
        np.asarray(rows_w, dtype=np.float64),
    )


def compute_anchor_spread(anchor_xy: np.ndarray) -> float:
    """Geometric spread of the anchor cloud in pixels.

    Defined as ``sqrt(var_x + var_y)`` which collapses to 0 for collinear
    points along *either* axis — it gives ``deformation_confidence`` what
    it needs to back off in degenerate geometries.
    """
    if anchor_xy.shape[0] < 2:
        return 0.0
    vx = float(np.var(anchor_xy[:, 0]))
    vy = float(np.var(anchor_xy[:, 1]))
    return math.sqrt(max(vx + vy, 0.0))


# ---------------------------------------------------------------------------
# Tier-0 cascade — easy-match committer (no kernel eval)
# ---------------------------------------------------------------------------

def commit_easy_matches(
    tracks: list,
    matched_tracks: set,
    matched_dets: set,
    all_candidates: list,
    colors_all,
    args,
    *,
    radius_frac: float = 1.0,
    colour_max_frac: float = 0.6,
    log_writer=None,
    frame_idx: int = 0,
):
    """Cheap unambiguous track→candidate commits, used to seed the field fit.

    For each currently-unmatched track, find the candidates within
    ``radius_frac * t.r`` of its current position.  A match is "easy" iff:
      * exactly one such (unmatched) candidate exists,
      * it is colour-compatible (BGR euclidean ≤ ``colour_max_frac * args.max_color``),
      * no other unmatched track also points to that candidate.

    Easy matches are committed in place (track.x/y/r/color/missed updated,
    matched_tracks and matched_dets sets mutated) using the same EMA colour
    rule the main MAIN-9 assignment path uses, so the resulting state is
    indistinguishable from a normal commit.

    Returns
    -------
    easy_anchors : list of (cur_x, cur_y, vx, vy)
        Current-frame anchor tuples for the field fit (vx, vy = cur − prev).
    n_committed : int
        How many tracks were easy-matched this frame.
    """
    if not tracks or not all_candidates:
        return [], 0
    from scipy.spatial import cKDTree

    cand_xy = np.array([(d[0], d[1]) for d in all_candidates], dtype=np.float64)
    if cand_xy.size == 0:
        return [], 0
    cand_kdt = cKDTree(cand_xy)

    colour_max = float(getattr(args, "max_color", 60.0)) * float(colour_max_frac)

    # [Phase 4 Stage 2c] Easy-pass hygiene (--easy-competition-check):
    # velocity-predicted positions of all unmatched tracks, for the
    # competition-at-comparable-prior check. Flag off -> never built.
    _hygiene = getattr(args, "easy_competition_check", 0) == 1
    _pred_kdt = None
    _pred_ti = None
    if _hygiene:
        _rows = []
        _pred_ti = []
        for ti, t in enumerate(tracks):
            if ti in matched_tracks:
                continue
            if t.px is not None and t.py is not None:
                _rows.append((t.x + (t.x - t.px), t.y + (t.y - t.py)))
            else:
                _rows.append((t.x, t.y))
            _pred_ti.append(ti)
        if _rows:
            _pred_kdt = cKDTree(np.asarray(_rows, dtype=np.float64))

    # For every unmatched track, find candidates inside radius_frac * t.r.
    candidate_claims: dict[int, list[int]] = {}    # cand_idx -> [track_idx, ...]
    track_pick: dict[int, int] = {}                # track_idx -> chosen cand_idx
    for ti, t in enumerate(tracks):
        if ti in matched_tracks:
            continue
        R = max(2.0, float(radius_frac) * float(t.r))
        close = cand_kdt.query_ball_point([t.x, t.y], r=R)
        close = [k for k in close if k not in matched_dets]
        if len(close) != 1:
            continue
        ck = close[0]
        # Colour compatibility check (BGR euclidean).
        det_bgr = np.asarray(colors_all[ck], dtype=np.float64)
        trk_bgr = np.asarray(t.color_bgr, dtype=np.float64)
        cdist = float(np.linalg.norm(det_bgr - trk_bgr))
        if cdist > colour_max:
            continue
        if _hygiene:
            cx_, cy_ = float(cand_xy[ck, 0]), float(cand_xy[ck, 1])
            # (ii) Displacement consistency: implied displacement deviating
            # > ~1 nn-spacing (~2r) from the track's own velocity prediction
            # demotes to hard — the plug-flow block failure mode (the only
            # det within 1r of the STALE position is the shifted neighbour).
            if t.px is not None and t.py is not None:
                vx_, vy_ = t.x - t.px, t.y - t.py
                dev = math.hypot((cx_ - t.x) - vx_, (cy_ - t.y) - vy_)
                if dev > 2.0 * float(t.r):
                    continue
            # (i) Competition at COMPARABLE prior: another unmatched track
            # whose velocity-predicted position is comparably close to this
            # det (within R and within 1.5x the claimant's own predicted
            # distance) -> not easy. The "comparable" ratio matters: any-
            # competitor-in-R was measured too strict on dense noisy scenes
            # (K01 s1 -0.010 HOTA from mutual disqualification of slow
            # neighbours).
            if _pred_kdt is not None:
                if t.px is not None and t.py is not None:
                    _own_pd = math.hypot(cx_ - (t.x + (t.x - t.px)),
                                         cy_ - (t.y + (t.y - t.py)))
                else:
                    _own_pd = math.hypot(cx_ - t.x, cy_ - t.y)
                _near = _pred_kdt.query_ball_point([cx_, cy_], r=R)
                _bar = max(1.5 * _own_pd, 2.0)
                _competed = False
                for h in _near:
                    if _pred_ti[h] == ti:
                        continue
                    hx, hy = _pred_kdt.data[h]
                    if math.hypot(cx_ - hx, cy_ - hy) <= _bar:
                        _competed = True
                        break
                if _competed:
                    continue
        track_pick[ti] = ck
        candidate_claims.setdefault(ck, []).append(ti)

    # Target uniqueness: only commit when exactly one track claims this candidate.
    easy_anchors = []
    n_committed = 0
    beta = float(getattr(args, "colour_ema", 0.3))
    if beta < 0.0: beta = 0.0
    elif beta > 1.0: beta = 1.0
    for ck, claimants in candidate_claims.items():
        if len(claimants) != 1:
            continue
        ti = claimants[0]
        if ti in matched_tracks or ck in matched_dets:
            continue   # paranoia: another path may have just claimed it
        t = tracks[ti]
        x, y, r, _score = all_candidates[ck]
        vx = float(x) - float(t.x)
        vy = float(y) - float(t.y)
        # Commit (mirrors the MAIN-9 update block).
        t.px, t.py, t.pr = t.x, t.y, t.r
        t.x, t.y, t.r = float(x), float(y), 0.85 * float(t.r) + 0.15 * float(r)
        new_c = np.asarray(colors_all[ck], dtype=np.float64)
        old_c = np.asarray(t.color_bgr, dtype=np.float64)
        t.color_bgr = tuple((beta * new_c + (1.0 - beta) * old_c).tolist())
        t.missed = 0
        t.last_det_pos = (float(x), float(y))
        t.last_det_r = float(r)
        matched_tracks.add(ti)
        matched_dets.add(ck)
        easy_anchors.append((float(x), float(y), vx, vy))
        n_committed += 1
        # [Phase 4 Stage 2c/3] Easy-commit audit row (same GT methodology
        # as cluster commits; population tag 'easy').
        if log_writer is not None:
            log_writer.writerow([frame_idx, "easy", t.id,
                                 f"{float(x):.2f}", f"{float(y):.2f}", "", "", 1])
    return easy_anchors, n_committed


# ---------------------------------------------------------------------------
# Phase 3 Stage B — displacement-adaptive cost weights
# ---------------------------------------------------------------------------

def observed_displacement_ratio(easy_matches: list, tracks: list) -> float:
    """Observed mean per-frame displacement over mean track radius.

    Source preference: this frame's easy-match displacements (fresh,
    current-frame observations); when fewer than 8 exist, fall back to
    the last-transition velocities of all tracks (steady-state correct).
    Returns 0.0 when nothing is observable (first frame) — the schedule
    then degrades to the static configuration.

    Driven by OBSERVATION, never the benchmark's --frame-step: the
    tracker must not know the benchmark exists, and observed displacement
    generalises to real low-frame-rate data.
    """
    if not tracks:
        return 0.0
    mean_r = float(np.mean([t.r for t in tracks]))
    if mean_r <= 0:
        return 0.0
    mags = [math.hypot(vx, vy) for (_x, _y, vx, vy) in easy_matches]
    if len(mags) < 8:
        mags = [math.hypot(t.x - t.px, t.y - t.py)
                for t in tracks if t.px is not None and t.py is not None]
    if not mags:
        return 0.0
    return float(np.mean(mags)) / mean_r


def adaptive_weight_schedule(
    disp_ratio: float,
    w_kernel_max: float,
    *,
    ratio_lo: float = 0.3,
    ratio_hi: float = 1.5,
    gate_scale_max: float = 3.0,
):
    """Map an observed displacement ratio to (w_kernel_eff, gate_scale).

    Smoothstep between ratio_lo (displacement ~ a third of a radius per
    frame: the static regime — kernel term off, gate unchanged) and
    ratio_hi (displacement >= 1.5 radii: full kernel weight + widest
    gate). On the cbtv corpus, step 1 sits near ratio ~0.2 (static),
    step 4 mid-ramp, step 8 saturated — but the input is the *observed*
    motion, so real low-frame-rate data lands wherever it lands.

    Returns
    -------
    w_kernel_eff : float in [0, w_kernel_max]
    gate_scale : float in [1, gate_scale_max] — multiplier for max_dist.
    """
    if ratio_hi <= ratio_lo:
        t = 1.0 if disp_ratio >= ratio_hi else 0.0
    else:
        t = (disp_ratio - ratio_lo) / (ratio_hi - ratio_lo)
        t = min(max(t, 0.0), 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    w_k = float(w_kernel_max) * smooth
    gate_scale = 1.0 + (gate_scale_max - 1.0) * smooth
    return w_k, gate_scale


class AdaptiveArgsView:
    """Read-only args proxy carrying per-frame effective cost weights.

    Instance attributes shadow the four adapted values; everything else
    delegates to the base args. Passed ONLY into compute_cost_matrix_gpu
    so no other consumer ever sees adapted values.
    """
    def __init__(self, base, w_dist, w_color, w_kernel, max_dist):
        self._base = base
        self.w_dist = w_dist
        self.w_color = w_color
        self.w_kernel = w_kernel
        self.max_dist = max_dist

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)


def adaptive_args_view(args, easy_matches: list, tracks: list):
    """Build the per-frame effective-args view for the cost matrix.

    Keeps the 0.7:0.3 w_dist:w_color convention at the effective kernel
    weight: w_dist = 0.7*(1-w_k_eff), w_color = 0.3*(1-w_k_eff).

    Returns (args_view, w_kernel_eff, disp_ratio).
    """
    ratio = observed_displacement_ratio(easy_matches, tracks)
    w_k_eff, gate_scale = adaptive_weight_schedule(
        ratio, float(getattr(args, "w_kernel", 0.0)))
    view = AdaptiveArgsView(
        args,
        w_dist=0.7 * (1.0 - w_k_eff),
        w_color=0.3 * (1.0 - w_k_eff),
        w_kernel=w_k_eff,
        max_dist=float(args.max_dist) * gate_scale,
    )
    return view, w_k_eff, ratio
