"""Pruned from the research module ``gpu_strategies.py``: only the code reachable from the shipped
detectors (csv, ws_multi_vote / ws_multi_vote_gpu, ws_vgl_log, cellpose) is kept, unchanged.

Original module docstring:

gpu_strategies.py - Phase 3d: GPU-accelerated detection strategies.

Implements three things:

  1. ``_classic_watershed_gpu(input_img, rmin, rmax)`` -- the GPU counterpart
     to ``detection_diagnostics.classic_watershed``. Numerically identical
     output (same Otsu threshold, same DT, same peak_local_max, same
     cv2.watershed, same per-basin centroid math). The only differences are:
       a) DT is computed on GPU via cupy (5-15x faster than scipy CPU)
       b) Per-basin centroid extraction is vectorized via np.bincount
          (~130x faster than the per-basin np.where loop in classic_watershed)
     (b) is allowed because the spec only forbids modifying the public CPU
     ``classic_watershed`` -- a private GPU helper that produces identical
     output is not a CPU-path change.

  2. ``ws_multi_vote_gpu(...)`` -- runs ``_classic_watershed_gpu`` on V, Gray,
     L channels and merges votes with the unchanged ``_merge_votes`` helper.

  3. ``ws_vgl_log_gpu(...)`` -- builds the VGL scalar on GPU, applies an
     FFT-domain Gaussian blur on GPU, applies a 3x3 Laplacian RawKernel on
     GPU, then runs ``_classic_watershed_gpu`` on the negated, clipped,
     renormalized response.

Public-API parity with the CPU strategies: same signature, same return
shape (detections, elapsed, diag, surface).
"""

import math
import time

import cv2
import numpy as np
from scipy.ndimage import label as ndimage_label
from skimage.feature import peak_local_max

from .gpu_backend import (
    GPU_AVAILABLE,
    HAS_CUPY,
    HAS_CUPY_DT,
    HAS_CUPY_RAWKERNEL,
    gpu_distance_transform_edt,
    gpu_laplacian_3x3,
    gpu_synchronize,
    to_cpu,
    to_gpu,
)


def _empty_result(method, elapsed, note=None):
    diag = {'method': method, 'empty': True}
    if note is not None:
        diag['note'] = note
    return ([], elapsed, diag, None)

def _extract_basins_vectorized(markers_ws, img_u8, rmin, rmax):
    """Extract one detection per watershed basin (label >= 2).

    Numerically identical to the per-basin loop in
    ``detection_diagnostics.classic_watershed`` (verified bit-exact: 0.000000 px
    centroid drift, same area / r_est / score). The advantage is purely
    speed: a 145-basin frame goes from ~300ms (np.where per basin) to
    ~2ms (single np.bincount over all valid pixels).
    """
    H, W = markers_ws.shape
    min_area = math.pi * (max(1.0, float(rmin) * 0.4)) ** 2
    max_area = math.pi * (max(2.0, float(rmax) * 2.5)) ** 2

    valid = markers_ws >= 2
    if not valid.any():
        return []

    labels_flat = markers_ws[valid]
    img_flat = img_u8[valid].astype(np.float64)
    ys_all, xs_all = np.nonzero(valid)
    unique_labels, inv = np.unique(labels_flat, return_inverse=True)
    K = unique_labels.size

    area = np.bincount(inv, minlength=K)
    intens_sum = np.bincount(inv, weights=img_flat, minlength=K)
    safe_intens = np.maximum(intens_sum, 1e-6)
    cx = np.bincount(inv, weights=xs_all * img_flat, minlength=K) / safe_intens
    cy = np.bincount(inv, weights=ys_all * img_flat, minlength=K) / safe_intens

    # Zero-intensity fallback: centroid -> arithmetic mean of pixel coords.
    zero_mask = intens_sum <= 1e-6
    if zero_mask.any():
        safe_area = np.maximum(area, 1)
        cx_fb = np.bincount(inv, weights=xs_all.astype(np.float64),
                             minlength=K) / safe_area
        cy_fb = np.bincount(inv, weights=ys_all.astype(np.float64),
                             minlength=K) / safe_area
        cx[zero_mask] = cx_fb[zero_mask]
        cy[zero_mask] = cy_fb[zero_mask]

    mean_intens = intens_sum / np.maximum(area, 1)
    keep = (area >= min_area) & (area <= max_area)
    r_est = np.sqrt(area / math.pi)
    score = mean_intens / 255.0

    out = []
    for i in range(K):
        if not keep[i]:
            continue
        out.append((float(cx[i]), float(cy[i]),
                    float(r_est[i]), float(score[i])))
    return out

def _classic_watershed_gpu(input_img, rmin=6, rmax=10):
    """GPU-accelerated counterpart of detection_diagnostics.classic_watershed.

    The pipeline is structurally identical:
      Otsu threshold (CPU) -> morph open/dilate (CPU) -> DT (GPU) ->
      peak_local_max (CPU) -> cv2.watershed (CPU) -> vectorized basin
      extraction (CPU).

    Returns (dets, diag, inter) with the same shape as classic_watershed.
    Falls back to CPU DT if the GPU isn't available; the public dispatchers
    in the strategy functions normally guard against that, but we keep the
    fallback here so this helper is self-contained.

    NOTE: this helper does NOT honor ``detection_benchmark.TOPHAT_THRESHOLD``
    because the Phase 3d GPU strategies are scoped to the un-tophatted
    (default) configuration. Adding tophat support is a follow-up.
    """
    t0 = time.perf_counter()
    img_u8 = (input_img if input_img.dtype == np.uint8
              else cv2.normalize(input_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    H, W = img_u8.shape[:2]

    # Otsu threshold + invert if >60% foreground (matches CPU).
    _, binary = cv2.threshold(img_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.sum() / 255 > 0.6 * H * W:
        binary = cv2.bitwise_not(binary)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=2)
    sure_bg = cv2.dilate(cleaned, k5, iterations=3)

    # Distance transform on GPU (the largest single CPU cost in this pipeline).
    if GPU_AVAILABLE and HAS_CUPY_DT:
        cleaned_gpu = to_gpu(cleaned > 0)
        dt_gpu = gpu_distance_transform_edt(cleaned_gpu)
        dt = to_cpu(dt_gpu).astype(np.float32)
    else:
        from scipy.ndimage import distance_transform_edt as _cpu_dt
        dt = _cpu_dt(cleaned).astype(np.float32)

    thr_val = (max(1.0, 0.35 * float(dt.max())) if dt.max() > 0 else 1.0)
    if dt.max() > 0:
        coords = peak_local_max(dt, min_distance=int(max(1, rmin)),
                                threshold_abs=thr_val)
    else:
        coords = np.zeros((0, 2), dtype=int)

    sure_fg_peaks = np.zeros_like(dt, dtype=np.uint8)
    for py, px in coords:
        sure_fg_peaks[py, px] = 255
    sure_fg = cv2.dilate(sure_fg_peaks, k3, iterations=1)

    unknown = cv2.subtract(sure_bg, sure_fg)
    labels_ccl, n_features = ndimage_label(sure_fg)
    markers = np.zeros((H, W), dtype=np.int32)
    if n_features > 0:
        ml = labels_ccl > 0
        markers[ml] = labels_ccl[ml] + 1
    markers[(sure_bg == 0)] = 1

    bgr_for_ws = (cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
                  if len(img_u8.shape) == 2 else img_u8.copy())
    markers_ws = cv2.watershed(bgr_for_ws, markers.copy())

    dets = _extract_basins_vectorized(markers_ws, img_u8, rmin, rmax)

    elapsed = time.perf_counter() - t0
    diag = {
        'n_features': n_features,
        'n_dets': len(dets),
        'binary_pct': float(binary.sum() / 255 / max(1, H * W)),
        'sure_fg_pct': float(sure_fg.sum() / 255 / max(1, H * W)),
        'time_ms': elapsed * 1000.0,
        'gpu': bool(GPU_AVAILABLE and HAS_CUPY_DT),
    }
    inter = {
        'binary': binary, 'cleaned': cleaned,
        'sure_bg': sure_bg, 'sure_fg': sure_fg, 'unknown': unknown,
        'dt': dt, 'markers_ws': markers_ws, 'peak_coords': coords,
    }
    return dets, diag, inter

def ws_multi_vote_gpu(bgr, channels, gray, rmin, rmax, topk=3,
                     _merge_votes=None):
    """GPU counterpart of ws_multi_vote. Same merge tolerance, same channels.

    The merge function is injected from strategy_combination_benchmark to
    avoid a circular import (this module is imported by that runner).
    """
    if _merge_votes is None:
        # Lazy import to avoid the cycle at module load time.
        from .strategies import _merge_votes as _mv
        _merge_votes = _mv

    t0 = time.perf_counter()
    try:
        dets_V, _, inter_V = _classic_watershed_gpu(channels['V'],
                                                     rmin=rmin, rmax=rmax)
        dets_G, _, _ = _classic_watershed_gpu(channels['Gray'],
                                              rmin=rmin, rmax=rmax)
        dets_L, _, _ = _classic_watershed_gpu(channels['L'],
                                              rmin=rmin, rmax=rmax)
        merged = _merge_votes([list(dets_V), list(dets_G), list(dets_L)],
                              tol=float(rmin))
    except Exception as e:
        return _empty_result('ws_multi_vote_gpu', time.perf_counter() - t0,
                             note=f'error:{e}')
    elapsed = time.perf_counter() - t0
    diag = {
        'method': 'ws_multi_vote_gpu',
        'n_V': len(dets_V), 'n_G': len(dets_G), 'n_L': len(dets_L),
        'n_merged': len(merged),
        'gpu': bool(GPU_AVAILABLE and HAS_CUPY_DT),
    }
    surface = inter_V.get('dt') if isinstance(inter_V, dict) else None
    return (merged, elapsed, diag, surface)

def _vgl_combo_cpu(channels):
    """Build the 0.5V + 0.3G + 0.2L scalar field on CPU as float64.

    Mirrors strategy_combination_benchmark._vgl_combo (we duplicate the
    body here rather than importing it to avoid a circular import).
    """
    V = channels['V'].astype(np.float64)
    G = channels['Gray'].astype(np.float64)
    if 'L' in channels:
        L = channels['L'].astype(np.float64)
    else:
        L = G
    return 0.5 * V + 0.3 * G + 0.2 * L

def ws_vgl_log_gpu(bgr, channels, gray, rmin, rmax, topk=3):
    """GPU counterpart of ws_vgl_log.

    Pipeline:
      VGL scalar (CPU)
        -> Gaussian blur on CPU via cv2.GaussianBlur (bit-exact match)
        -> Laplacian on GPU via the 3x3 RawKernel matching cv2.Laplacian
        -> negate, clip>=0, normalize, cast u8 (GPU)
        -> _classic_watershed_gpu (Otsu+morph CPU, DT GPU, watershed CPU,
           vectorised basin extraction CPU)

    Why CPU GaussianBlur instead of FFT-based GPU GaussianBlur:
    The FFT path produces a smoothed image within 0.006 abs of cv2's
    spatial Gaussian (well within the 1-LSB intermediate tolerance), but
    the downstream negated-Laplacian -> clip -> watershed pipeline
    amplifies LSB-level noise into bifurcating basin choices on dense
    frames, causing detection drift > 0.5 px on roughly 1-4% of basins.
    Doing GaussianBlur on CPU (~13 ms) lets us keep the intermediate
    bit-identical to the CPU strategy while still getting ~10x overall
    speedup from GPU Laplacian + GPU DT + vectorised basin extraction.
    """
    if not (HAS_CUPY and HAS_CUPY_RAWKERNEL):
        # Without RawKernel we cannot run the GPU Laplacian.
        return _empty_result('ws_vgl_log_gpu', 0.0, note='gpu_unavailable')

    t0 = time.perf_counter()
    try:
        sigma = float(rmin) / float(np.sqrt(2.0))
        # CPU: build VGL scalar and Gaussian-blur it (bit-exact vs cv2 path).
        scalar_cpu = _vgl_combo_cpu(channels)
        smoothed_cpu = cv2.GaussianBlur(scalar_cpu, (0, 0), sigmaX=sigma)
        # GPU: upload, take 3x3 negated Laplacian, clip, normalize, cast u8.
        import cupy as cp
        smoothed_gpu = to_gpu(smoothed_cpu)
        lap_gpu = gpu_laplacian_3x3(smoothed_gpu)
        response_gpu = -lap_gpu
        response_gpu = cp.clip(response_gpu, 0.0, None)
        peak = float(response_gpu.max())
        if peak < 1e-6:
            response_u8 = np.zeros(smoothed_cpu.shape, dtype=np.uint8)
        else:
            response_u8_gpu = (255.0 * response_gpu / peak).astype(cp.uint8)
            response_u8 = to_cpu(response_u8_gpu)
        gpu_synchronize()

        dets, diag, inter = _classic_watershed_gpu(response_u8,
                                                    rmin=rmin, rmax=rmax)
    except Exception as e:
        return _empty_result('ws_vgl_log_gpu', time.perf_counter() - t0,
                             note=f'error:{e}')
    elapsed = time.perf_counter() - t0
    out_diag = {
        'method': 'ws_vgl_log_gpu',
        'n_raw': len(dets), 'sigma': sigma,
        'gpu': True,
    }
    if isinstance(diag, dict):
        out_diag.update({f'ws_{k}': v for k, v in diag.items() if np.isscalar(v)})
    surface = inter.get('dt') if isinstance(inter, dict) else None
    return (list(dets), elapsed, out_diag, surface)
