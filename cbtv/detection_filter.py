"""Multi-stage detection filtering, candidate capping, peak maps."""
import math
from typing import Any

import cv2
import numpy as np

from . import gpu
from .correlation import nms_mask, peaks_to_list
from .gpu import cp, to_gpu, to_numpy


def _multichannel_stats(channels_dict, ksz: int):
    """
    Compute local stats across multiple channels (L, S, V, B, Gray) matching detector behavior.
    Returns fused variance, gradient, and per-channel gradients for circularity checks.
    """
    H, W = channels_dict[list(channels_dict.keys())[0]].shape

    # Storage for per-channel stats
    vars_list = []
    grads_list = []
    gx_channels = {}
    gy_channels = {}

    # Compute stats for each channel
    for name, chan in channels_dict.items():
        g = chan.astype(np.float32)

        # Variance
        mean = cv2.blur(g, (ksz, ksz))
        mean2 = cv2.blur(g * g, (ksz, ksz))

        if gpu.USE_CUPY:
            try:
                mean_gpu = to_gpu(mean)
                mean2_gpu = to_gpu(mean2)
                var = cp.maximum(mean2_gpu - mean_gpu * mean_gpu, 0)
                var = to_numpy(var)
            except Exception:
                var = np.maximum(mean2 - mean * mean, 0)
        else:
            var = np.maximum(mean2 - mean * mean, 0)

        # Gradient
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

        if gpu.USE_CUPY:
            try:
                gx_gpu = to_gpu(gx)
                gy_gpu = to_gpu(gy)
                grad = cp.sqrt(gx_gpu * gx_gpu + gy_gpu * gy_gpu)
                grad = to_numpy(grad)
            except Exception:
                grad = np.sqrt(gx * gx + gy * gy)
        else:
            grad = np.sqrt(gx * gx + gy * gy)

        gmax = cv2.blur(grad, (ksz, ksz))

        vars_list.append(var)
        grads_list.append(gmax)
        gx_channels[name] = gx
        gy_channels[name] = gy

    # Fuse across channels: use weighted average with boosted importance for reliable channels
    # This balances between being too permissive (maximum) and too aggressive (median)
    channel_names = list(channels_dict.keys())
    weights: Any = []                 # list, then the normalised float array below (T041: mypy)
    for name in channel_names:
        # Boost weight for most reliable channels (L and Gray)
        if name == 'L': weights.append(0.35)      # LAB lightness - most important
        elif name == 'Gray': weights.append(0.35) # Grayscale - most important
        elif name == 'V': weights.append(0.15)    # HSV value - moderate
        elif name == 'B': weights.append(0.10)    # YCrCb brightness - lower
        elif name == 'S': weights.append(0.05)    # HSV saturation - least reliable
        else: weights.append(0.05)                # fallback

    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Weighted average fusion
    vars_stack = np.stack(vars_list, axis=0)
    grads_stack = np.stack(grads_list, axis=0)

    if gpu.USE_CUPY:
        try:
            vars_gpu = to_gpu(vars_stack)
            grads_gpu = to_gpu(grads_stack)
            weights_gpu = to_gpu(weights.reshape(-1, 1, 1))
            var_fused = to_numpy(cp.sum(vars_gpu * weights_gpu, axis=0))
            grad_fused = to_numpy(cp.sum(grads_gpu * weights_gpu, axis=0))
        except Exception:
            vars_stack_np = vars_stack
            grads_stack_np = grads_stack
            weights_np = weights.reshape(-1, 1, 1)
            var_fused = np.sum(vars_stack_np * weights_np, axis=0)
            grad_fused = np.sum(grads_stack_np * weights_np, axis=0)
    else:
        weights_np = weights.reshape(-1, 1, 1)
        var_fused = np.sum(vars_stack * weights_np, axis=0)
        grad_fused = np.sum(grads_stack * weights_np, axis=0)

    return var_fused, grad_fused, gx_channels, gy_channels

def _local_stats(gray_u8: np.ndarray, ksz: int):
    """Legacy single-channel stats function (kept for compatibility)"""
    g = gray_u8.astype(np.float32)
    mean = cv2.blur(g, (ksz, ksz))
    mean2 = cv2.blur(g * g, (ksz, ksz))

    if gpu.USE_CUPY:
        try:
            mean_gpu = to_gpu(mean)
            mean2_gpu = to_gpu(mean2)
            var = cp.maximum(mean2_gpu - mean_gpu * mean_gpu, 0)
            var = to_numpy(var)
        except Exception:
            var = np.maximum(mean2 - mean * mean, 0)
    else:
        var = np.maximum(mean2 - mean * mean, 0)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    if gpu.USE_CUPY:
        try:
            gx_gpu = to_gpu(gx)
            gy_gpu = to_gpu(gy)
            grad = cp.sqrt(gx_gpu * gx_gpu + gy_gpu * gy_gpu)
            grad = to_numpy(grad)
        except Exception:
            grad = np.sqrt(gx * gx + gy * gy)
    else:
        grad = np.sqrt(gx * gx + gy * gy)

    gmax = cv2.blur(grad, (ksz, ksz))
    return var, gmax, gx, gy

def _estimate_local_density(xi, yi, all_dets, search_radius=50.0):
    """
    Estimate local particle density around point (xi, yi).
    Returns number of detections within search_radius.
    """
    count = 0
    for x, y, r, s in all_dets:
        dist = math.hypot(x - xi, y - yi)
        if dist <= search_radius:
            count += 1
    return count

def filter_detections_multistage(gray_u8, dets, hard=None, soft=None, channels_dict=None, args=None, stats=None, small_image_mode=False):
    """
    Filter detections using multi-channel information matching detector behavior.
    If channels_dict is provided, uses multi-channel filtering (L, S, V, B, Gray).
    Otherwise falls back to single-channel grayscale filtering.

    Implements density-aware adaptive thresholding: in dense regions, we relax
    thresholds slightly to avoid over-filtering real particles.

    Pipeline diagram: see docs/architecture/PIPELINE_FLOWCHARTS.md
    Flowchart 11 (Filter Pipeline Detail). Section labels [FILT-1..7]
    map 1:1 to that diagram's stages.

    [Phase 4B] Phase 4B added: resolution scaling via res_scale = min(H,W)/1024
    so all pixel-based thresholds (edge_distance, density search radius) adapt
    to non-1024 videos; density-adaptive threshold relaxation; sparse filter
    tightening when total candidates < 50.
    [Phase 5-G2] Phase 5-G2 added: obstacle mask post-filter (rejecting
    detections inside masked obstacle regions) — applied by the caller after
    this function returns.

    [Decision Rationale] The 'stats' parameter allows the caller (typically
    user_detection_with_bayes_adapt) to pre-compute multichannel local stats
    once and reuse them here, avoiding a duplicate _multichannel_stats call.
    When stats is None, this function computes them itself from channels_dict
    or gray_u8 as a fallback.
    """
    if not dets: return [], [], []
    H, W = gray_u8.shape

    radii = [r for _,_,r,_ in dets]
    if gpu.USE_CUPY:
        try:
            radii_gpu = cp.array(radii, dtype=cp.float32)
            med_r = float(cp.median(radii_gpu))
            avg_r = float(cp.mean(radii_gpu))
        except Exception:
            radii = np.array(radii, np.float32)
            med_r = float(np.median(radii))
            avg_r = float(np.mean(radii))
    else:
        radii = np.array(radii, np.float32)
        med_r = float(np.median(radii))
        avg_r = float(np.mean(radii))

    ksz = int(max(3, 2*avg_r+1)); ksz += (ksz % 2 == 0)

    if stats is not None:
        var, gmax, gx, gy = stats
    else:
        # Use multi-channel stats if available, otherwise fall back to grayscale
        if channels_dict is not None:
            var, gmax, gx_channels, gy_channels = _multichannel_stats(channels_dict, ksz)
            # For circularity, use the channel with strongest gradient (best signal)
            # Build combined gx, gy from strongest gradient per pixel
            gx_list = [gx_channels[name] for name in gx_channels.keys()]
            gy_list = [gy_channels[name] for name in gy_channels.keys()]
            gx_stack = np.stack(gx_list, axis=0)
            gy_stack = np.stack(gy_list, axis=0)
            # Pick gradient from channel with maximum magnitude at each pixel
            grad_mags = np.sqrt(gx_stack**2 + gy_stack**2)
            max_idx = np.argmax(grad_mags, axis=0)
            gx = np.take_along_axis(gx_stack, max_idx[None, :, :], axis=0)[0]
            gy = np.take_along_axis(gy_stack, max_idx[None, :, :], axis=0)[0]
        else:
            var, gmax, gx, gy = _local_stats(gray_u8, ksz)

    # Resolution-adaptive scaling: scale pixel-based thresholds by min(H,W)/1024
    ref_dim = 1024.0
    res_scale = min(H, W) / ref_dim

    # [Decision Rationale] Filter thresholds tuned for 1024x1024 primary video (see Flowchart 11 — Filter Pipeline):
    # - edge_distance=3: particles within 3px of frame edge have clipped kernels, unreliable NCC
    # - min_local_variance=47.0: below this, region is too uniform to contain a real particle
    #   (background noise has var ~20-40; real particles have var 60+)
    # - min_gradient_strength=4.75: rejects flat/blurry false detections (real particles have edges)
    # - min_circularity=0.50: particles are spherical; lower means >50% of radial samples match
    # - intensity_extreme: 8/247 reject near-black/near-white artifacts (sensor noise, glare)
    # Soft thresholds are stricter (higher variance, gradient, circularity) — detections that pass
    # soft but not hard can be matched to existing tracks but cannot spawn new ones.
    hard = dict(edge_distance=3, max_radius_factor=1.65,
                min_local_variance=47.0,
                min_gradient_strength=4.75, min_circularity=0.50,
                intensity_uniform_std=14.0, intensity_extreme_low=8,
                intensity_extreme_high=247) if hard is None else hard
    soft = dict(max_radius_factor=1.55, min_local_variance=80.0,
                min_gradient_strength=5.6, min_circularity=0.70) if soft is None else soft
    # Apply resolution scaling to pixel-based thresholds (works for both default and provided)
    if res_scale < 0.95:  # Only scale for non-standard (smaller) resolutions
        hard["edge_distance"] = max(1, int(hard["edge_distance"] * res_scale))

    # Density-aware scaling parameters
    # Expected density for dense particle flow (based on ~460 particles in 1024x1024 frame)
    # With 50px search radius, expect ~10-15 particles in dense regions
    target_density = max(3.0, 12.0 * res_scale * res_scale)
    density_search_radius = max(10.0, 50.0 * res_scale)
    # Scale factor for threshold relaxation in dense regions
    density_scale_factor = 0.30  # Relax thresholds by up to 30% in densest regions

    kept, soft_keep, killed = [], [], []
    n_samples = 16
    ang = np.linspace(0, 2*np.pi, n_samples, endpoint=False)

    for x, y, r, s in dets:
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            continue

        # === [FILT-3] Density-Adaptive Threshold Relaxation (see Flowchart 11 — Stage 3) ===
        # [Phase 4B] Optional density-aware threshold scaling (controlled by flag)
        if args is not None and hasattr(args, 'use_density_adaptive_thresholds') and args.use_density_adaptive_thresholds:
            local_density = _estimate_local_density(xi, yi, dets, density_search_radius)
            density_ratio = min(local_density / target_density, 1.5)
            threshold_multiplier = 1.0 - (density_scale_factor * (density_ratio - 0.5))
            threshold_multiplier = max(0.70, min(1.0, threshold_multiplier))
        else:
            threshold_multiplier = 1.0  # Fixed threshold (faster)

        # Apply density-aware scaling to hard thresholds
        hard_var_thr = hard["min_local_variance"] * threshold_multiplier
        hard_grad_thr = hard["min_gradient_strength"] * threshold_multiplier
        hard_circ_thr = hard["min_circularity"] * threshold_multiplier

        # === [FILT-1] Edge Distance Filter (see Flowchart 11 — Stage 1) ===
        cond_edge = (xi < hard["edge_distance"] or yi < hard["edge_distance"]
                     or xi >= W - hard["edge_distance"] or yi >= H - hard["edge_distance"])
        # === [FILT-4] Variance / gradient hard gate (see Flowchart 11 — Stage 4) ===
        cond_big = r > med_r * hard["max_radius_factor"]
        cond_var = var[yi, xi] < hard_var_thr
        cond_g = gmax[yi, xi] < hard_grad_thr

        sx = xi + r * np.cos(ang); sy = yi + r * np.sin(ang)
        ok = (sx >= 1) & (sx < W - 1) & (sy >= 1) & (sy < H - 1)
        circ_ok = True
        if np.sum(ok) >= n_samples * 0.75:
            si = sx[ok].astype(int); sj = sy[ok].astype(int)
            gx_s = gx[sj, si]; gy_s = gy[sj, si]
            dx = (si - xi).astype(np.float32); dy = (sj - yi).astype(np.float32)

            nrm = np.sqrt(dx*dx + dy*dy) + 1e-6; dx /= nrm; dy /= nrm
            gn = np.sqrt(gx_s*gx_s + gy_s*gy_s) + 1e-6
            gx_n = gx_s / gn; gy_n = gy_s / gn
            align = np.abs(gx_n * dx + gy_n * dy).mean()
            circ_ok = align >= hard_circ_thr

        # === [FILT-2] Intensity Extreme Filter (see Flowchart 11 — Stage 2) ===
        # Reject regions that are extremely dark/bright AND nearly uniform
        # (sensor noise, glare, or saturation artefacts).
        y0, y1 = max(0, yi-int(r)), min(H, yi+int(r)+1)
        x0, x1 = max(0, xi-int(r)), min(W, xi+int(r)+1)
        roi = gray_u8[int(y0):int(y1), int(x0):int(x1)]
        intens_ok = True
        if roi.size:
            m = float(np.mean(roi))
            sd = float(np.std(roi))
            intens_ok = not (sd < hard["intensity_uniform_std"] and
                           (m < hard["intensity_extreme_low"] or m > hard["intensity_extreme_high"]))

        if cond_edge or cond_big or cond_var or cond_g or (not circ_ok) or (not intens_ok):
            killed.append((x, y, r, s)); continue

        # === [FILT-5] Soft Classification Path (see Flowchart 11 — Soft) ===
        # Detections that fail the hard gate but pass the (stricter) soft thresholds
        # become "soft" matches: they can be paired with existing tracks but
        # cannot spawn new tracks. This protects recall in marginal cases without
        # adding spurious new IDs.
        # Apply density-aware scaling to soft thresholds as well
        soft_var_thr = soft["min_local_variance"] * threshold_multiplier
        soft_grad_thr = soft["min_gradient_strength"] * threshold_multiplier
        soft_circ_thr = soft["min_circularity"] * threshold_multiplier

        cond_big_s  = r > med_r * soft["max_radius_factor"]
        cond_var_s  = var[yi, xi] < soft_var_thr
        cond_g_s    = gmax[yi, xi] < soft_grad_thr
        circ_ok_s = True
        if np.sum(ok) >= n_samples * 0.75:
            si = sx[ok].astype(int); sj = sy[ok].astype(int)
            gx_s = gx[sj, si]; gy_s = gy[sj, si]
            dx = (si - xi).astype(np.float32); dy = (sj - yi).astype(np.float32)

            nrm = np.sqrt(dx*dx + dy*dy) + 1e-6; dx /= nrm; dy /= nrm
            gn = np.sqrt(gx_s*gx_s + gy_s*gy_s) + 1e-6
            gx_n = gx_s / gn; gy_n = gy_s / gn
            align = np.abs(gx_n * dx + gy_n * dy).mean()
            circ_ok_s = align >= soft_circ_thr

        if cond_big_s or cond_var_s or cond_g_s or (not circ_ok_s):
            soft_keep.append((x, y, r, s))
        else:
            kept.append((x, y, r, s))

    # === [FILT-6] Sparse Filter Tightening (see Flowchart 11 — Sparse) ===
    # [Phase 4B] Sparse-scene FP tightening: when very few particles detected, re-filter
    # with stricter thresholds to reject background noise. Skipped in
    # small_image_mode (where the scene legitimately has few particles).
    total_candidates = len(kept) + len(soft_keep)
    if small_image_mode and total_candidates < 50:
        # Scene genuinely has few particles — don't apply tighter thresholds
        if total_candidates > 0:
            print(f"  [SMALL-IMAGE] Skipping sparse filter ({total_candidates} candidates)")
    elif total_candidates > 0 and total_candidates < 50:
        tighter_var = hard["min_local_variance"] * 1.3
        tighter_grad = hard["min_gradient_strength"] * 1.2
        tighter_circ = max(hard["min_circularity"], 0.60)
        kept2 = []
        for x, y, r, s in kept:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                if var[yi, xi] < tighter_var or gmax[yi, xi] < tighter_grad:
                    soft_keep.append((x, y, r, s))
                    continue
                # Re-check circularity with tighter threshold
                sx2 = xi + r * np.cos(ang); sy2 = yi + r * np.sin(ang)
                ok2 = (sx2 >= 1) & (sx2 < W - 1) & (sy2 >= 1) & (sy2 < H - 1)
                if np.sum(ok2) >= n_samples * 0.75:
                    si2 = sx2[ok2].astype(int); sj2 = sy2[ok2].astype(int)
                    gx_s2 = gx[sj2, si2]; gy_s2 = gy[sj2, si2]
                    dx2 = (si2 - xi).astype(np.float32); dy2 = (sj2 - yi).astype(np.float32)
                    nrm2 = np.sqrt(dx2*dx2 + dy2*dy2) + 1e-6; dx2 /= nrm2; dy2 /= nrm2
                    gn2 = np.sqrt(gx_s2*gx_s2 + gy_s2*gy_s2) + 1e-6
                    align2 = np.abs((gx_s2/gn2)*dx2 + (gy_s2/gn2)*dy2).mean()
                    if align2 < tighter_circ:
                        soft_keep.append((x, y, r, s))
                        continue
            kept2.append((x, y, r, s))
        if len(kept2) < len(kept):
            print(f"[SPARSE] Tightened: {len(kept)} -> {len(kept2)} kept "
                  f"(total_candidates={total_candidates})")
        kept = kept2

    # === [FILT-7] Obstacle Mask Safety-Net (see Flowchart 11 — Obstacle) ===
    # [Phase 5-G2] Note: the obstacle mask filter itself is applied by the caller
    # (typically `main()` after this function returns), using args.obstacle_mask
    # as a boolean array. The mask is intentionally kept outside this function
    # so it can also be applied to detections that bypass multi-stage filtering
    # (e.g. recovery candidates). The K06 mask is loaded with erode=8 by default
    # (see Phase 5-K06).

    return kept, soft_keep, killed

def cap_candidates(kept_sorted, soft_list, prev_tracks, mode="warmup", cap_min=40, cap_warmup=8, extra_frac=0.15, fixed=500, ratio=1.0, frame_idx=0):
    """Cap detection candidates to prevent the Hungarian solver from being overwhelmed.

    Pipeline context:
        Called by: filter_detections_multistage (end of filtering pipeline),
                   user_detection_with_bayes_adapt
        Stage: filtering

    Excess candidates beyond the cap are demoted to soft_list rather than
    discarded, so they can still be recovered by the tracker if needed.

    Modes:
        "off"    — no cap, return inputs unchanged.
        "fixed"  — hard cap at `fixed` candidates regardless of track count.
        "ratio"  — cap at ceil(prev_tracks * ratio); floored to cap_min
                   during warmup frames.
        "warmup" — (default) adaptive cap: ceil(prev_tracks * (1 + extra_frac)),
                   floored to cap_min for the first cap_warmup frames.

    Args:
        kept_sorted: Candidate detections sorted by descending score
                    [list[tuple[float, float, float, float]]]
        soft_list:   Current soft (low-confidence) detection list; overflow
                    candidates are appended here [list[tuple]]
        prev_tracks: Number of active tracks from the previous frame [int]
        mode:        Capping strategy ("off", "fixed", "ratio", "warmup")
                    [str, default "warmup"]
        cap_min:     Minimum cap during warmup frames [int, default 40]
        cap_warmup:  Number of initial frames treated as warmup [int, default 8]
        extra_frac:  Fractional headroom above prev_tracks in "warmup" mode
                    [float, default 0.15]
        fixed:       Hard cap value used in "fixed" mode [int, default 500]
        ratio:       Multiplier applied to prev_tracks in "ratio" mode
                    [float, default 1.0]
        frame_idx:   Current frame index, used to detect warmup period
                    [int, default 0]

    Returns:
        Tuple of (kept, soft_list) where kept is the capped candidate list and
        soft_list is the updated overflow list
        [tuple[list, list]]
    """
    if mode=="off":
        return kept_sorted, soft_list
    if mode=="fixed":
        K = int(fixed)
        return kept_sorted[:K], soft_list + kept_sorted[K:]
    if mode=="ratio":
        base = int(math.ceil(prev_tracks * max(0.0, float(ratio))))
        if frame_idx < cap_warmup:
            base = max(base, cap_min)
        return kept_sorted[:base], soft_list + kept_sorted[base:]
    prev = int(prev_tracks)
    K_prev = prev + int(math.ceil(extra_frac * max(1, prev)))
    K = max(cap_min if frame_idx < cap_warmup else 1, K_prev)
    return kept_sorted[:K], soft_list + kept_sorted[K:]

def local_stats_map(C, ksize):
    """Compute local mean and standard deviation maps for adaptive peak thresholding.

    Pipeline context:
        Called by: peaks_from_map
        Stage: correlation

    Uses box-filter approximations (cv2.blur) to compute E[X] and E[X^2]
    within each (ksize x ksize) neighbourhood, then derives variance as
    E[X^2] - E[X]^2.  The standard deviation is clamped to a minimum of
    sqrt(1e-8) to avoid division by zero in subsequent z-score calculations.
    GPU acceleration via CuPy is used when available.

    Args:
        C:     2-D correlation response map [np.ndarray, float32, HxW]
        ksize: Neighbourhood size in pixels; forced to an odd number >= 3
               [int]

    Returns:
        Tuple of (mean_map, std_map), both the same shape as C
        [tuple[np.ndarray, np.ndarray]]
    """
    if ksize < 3: ksize = 3
    if ksize % 2 == 0: ksize += 1
    m = cv2.blur(C, (ksize, ksize))
    m2 = cv2.blur(C*C, (ksize, ksize))

    if gpu.USE_CUPY:
        m_gpu = to_gpu(m)
        m2_gpu = to_gpu(m2)
        v = cp.maximum(m2_gpu - m_gpu*m_gpu, 1e-8)
        s = cp.sqrt(v)
        return to_numpy(m), to_numpy(s)
    else:
        v = np.maximum(m2 - m*m, 1e-8)
        s = np.sqrt(v)
        return m, s

def peaks_from_map(C, r, thr_abs, nms_rad, mode="hybrid", local_k=2.5, local_ksize=None, topk_per_tile=0, tile_factor=3.0, enable_subpixel=False):
    """Extract detection peaks from a correlation map using hybrid absolute+local thresholding with NMS.

    Pipeline context:
        Called by: user_detection_with_bayes_adapt (main detection extraction step)
        Stage: correlation

    Three thresholding modes are supported:

    - "abs"    — keep only pixels that exceed thr_abs.
    - "local"  — keep pixels whose local z-score >= local_k (adaptive to scene
                 brightness variation).
    - "hybrid" — (default) union of "abs" and "local"; any pixel passing
                 either criterion is kept.

    All surviving pixels are additionally filtered by NMS (via nms_mask) so
    only local maxima within nms_rad pixels are retained.  An optional tiled
    top-K pass can supplement the mask with the highest-scoring peak in each
    spatial tile.

    Args:
        C:              2-D NCC correlation response map [np.ndarray, float32, HxW]
        r:              Particle radius in pixels; used to size the local stats
                       window when local_ksize is None [float]
        thr_abs:        Absolute NCC score threshold for "abs"/"hybrid" modes
                       [float]
        nms_rad:        NMS half-window radius in pixels (passed to nms_mask)
                       [int]
        mode:           Thresholding mode: "abs", "local", or "hybrid"
                       [str, default "hybrid"]
        local_k:        Z-score threshold for "local"/"hybrid" modes [float,
                       default 2.5]
        local_ksize:    Override neighbourhood size for local stats; defaults
                       to max(5, round(2*r)|1) [int | None, default None]
        topk_per_tile:  If > 0, additionally mark the top-K NMS peaks in each
                       spatial tile of size ~tile_factor*(2r+1) [int, default 0]
        tile_factor:    Controls tile size relative to particle diameter when
                       topk_per_tile is active [float, default 3.0]
        enable_subpixel: Forward to peaks_to_list for sub-pixel refinement
                        [bool, default False]

    Returns:
        List of (x, y, r, score) detections at surviving peak locations
        [list[tuple[float, float, float, float]]]
    """
    H,W = C.shape[:2]
    if local_ksize is None:
        local_ksize = max(5, int(round(2.0*r))|1)

    mask = nms_mask(C, nms=nms_rad)
    keep = (C >= float(thr_abs))

    if mode in ("local","hybrid"):
        mu, sig = local_stats_map(C, local_ksize)
        if gpu.USE_CUPY:
            C_gpu = to_gpu(C)
            mu_gpu = to_gpu(mu)
            sig_gpu = to_gpu(sig)
            z = (C_gpu - mu_gpu) / (sig_gpu + 1e-6)
            z = to_numpy(z)
        else:
            z = (C - mu) / (sig + 1e-6)
        keep = (keep | (z >= float(local_k)))
    mask = (mask & keep)

    if topk_per_tile and topk_per_tile > 0:
        tile = max(2*nms_rad+1, int(round(tile_factor * (2*r+1))))
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1 = min(H, y0+tile); x1 = min(W, x0+tile)
                sub = C[y0:y1, x0:x1]
                submask = nms_mask(sub, nms=nms_rad)
                ys, xs = np.where(submask)
                if len(ys)==0: continue
                vals = sub[ys, xs]
                order = np.argsort(-vals, kind="stable")[:int(topk_per_tile)]
                mask[y0 + ys[order], x0 + xs[order]] = True

    return peaks_to_list(C, mask, r, enable_subpixel=enable_subpixel)

