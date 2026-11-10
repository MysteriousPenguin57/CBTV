"""Particle colour extraction & GPU colour-distance matrices."""
import cv2
import numpy as np

from . import gpu
from .gpu import cp, to_gpu, to_numpy


def batch_compute_colors_gpu(frame_bgr, detections, use_gpu=True):
    """
    Vectorized color extraction for all detections.
    Instead of creating full HxW masks per particle (which takes gigabytes of memory
    and 2.3 seconds), we use localized crops which takes ~5ms.

    Args:
        frame_bgr: BGR frame (H, W, 3)
        detections: List of (x, y, r, score) tuples
        use_gpu: Whether to use GPU acceleration (not needed for this optimized version)

    Returns:
        List of (b, g, r) tuples (same order as detections)
    """
    if not detections:
        return []

    H, W = frame_bgr.shape[:2]
    colors = []

    for (x, y, r, _) in detections:
        cx, cy = int(round(x)), int(round(y))
        rad = int(max(2, round(r*1.15)))
        x0, y0 = max(0, cx-rad), max(0, cy-rad)
        x1, y1 = min(W, cx+rad+1), min(H, cy+rad+1)

        if x1 <= x0 or y1 <= y0:
            colors.append((0.0, 0.0, 0.0))
            continue

        crop = frame_bgr[y0:y1, x0:x1]

        # Localized circular mask
        Y, X = np.ogrid[:y1-y0, :x1-x0]
        dist_sq = (X - (cx - x0))**2 + (Y - (cy - y0))**2
        mask = (dist_sq <= rad**2).astype(np.uint8)

        # mean output is (B, G, R, alpha)
        mb, mg, mr, _ = cv2.mean(crop, mask=mask)
        colors.append((mb, mg, mr))

    return colors

def sample_weighted_colors(frame_bgr, dets, glare_threshold=220):
    """Saturation-weighted LAB colour + mean saturation at each (x, y, r, _)
    position of ``frame_bgr`` -- the same statistic the colour-distance
    matrix samples (CPU path). Used by --color-source track to store a
    track's colour at the moment it is matched. Returns (lab (n,3), S (n,))."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, W = frame_bgr.shape[:2]
    out_lab = np.zeros((len(dets), 3), np.float64)
    out_S = np.zeros(len(dets), np.float64)
    for k, (x, y, r, _) in enumerate(dets):
        cx, cy = int(round(x)), int(round(y))
        rad = int(max(2, round(r * 1.15)))
        x0, y0 = max(0, cx - rad), max(0, cy - rad)
        x1, y1 = min(W, cx + rad + 1), min(H, cy + rad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        Y, X = np.ogrid[:y1 - y0, :x1 - x0]
        mask = (X - (cx - x0)) ** 2 + (Y - (cy - y0)) ** 2 <= rad ** 2
        lab_vals = lab[y0:y1, x0:x1][mask]
        S_vals = hsv[y0:y1, x0:x1][mask][:, 1]
        if len(lab_vals) == 0:
            continue
        ng = (lab_vals[:, 0] < glare_threshold) | (S_vals > 30)
        if ng.sum() > 0:
            lab_vals, S_vals = lab_vals[ng], S_vals[ng]
        if S_vals.sum() > 0:
            out_lab[k] = np.average(lab_vals, axis=0, weights=S_vals / S_vals.sum())
        else:
            out_lab[k] = lab_vals.mean(axis=0)
        out_S[k] = float(S_vals.mean())
    return out_lab, out_S


def batch_smart_color_distance_gpu(track_colors, det_colors, frame_bgr, track_dets, det_list,
                                    saturation_weight=True, nonlinear_k=2.0, glare_threshold=220, use_gpu=True,
                                    track_lab_S=None):
    """
    Vectorized smart color distance computation with saturation weighting

    Args:
        track_colors: Array (M, 3) of track colors in BGR
        det_colors: Array (N, 3) of detection colors in BGR
        frame_bgr: Current frame
        track_dets: Array (M, 4) of track positions [(x, y, r, _), ...]
        det_list: Array (N, 4) of detection positions
        saturation_weight: Whether to use saturation weighting
        nonlinear_k: Non-linear penalty factor
        glare_threshold: Threshold for glare detection
        use_gpu: Whether to use GPU

    Returns:
        distance_matrix: (M, N) array of color distances
        confidence_matrix: (M, N) array of saturation confidences

    track_lab_S: optional (lab (M,3), S (M,)) stored track colours
        (--color-source track). When given, the track side is NOT
        re-sampled from the current frame at the track's old position.
    """
    M = len(track_colors)
    N = len(det_colors)

    if M == 0 or N == 0:
        return np.zeros((M, N), np.float32), np.ones((M, N), np.float32)

    # Try GPU path
    if use_gpu and gpu.USE_CUPY:
        try:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

            lab_gpu = to_gpu(lab)
            hsv_gpu = to_gpu(hsv)

            def get_weighted_color_batch_gpu(det_array):
                """Batch extract saturation-weighted LAB colors for all detections (GPU path)."""
                if len(det_array) == 0:
                    return cp.empty((0,3), dtype=cp.float32), cp.empty((0,), dtype=cp.float32)

                H, W = frame_bgr.shape[:2]

                cx = cp.asarray([round(x) for x, y, r, _ in det_array], dtype=cp.int32)
                cy = cp.asarray([round(y) for x, y, r, _ in det_array], dtype=cp.int32)
                rads = cp.asarray([max(2, round(r*1.15)) for x, y, r, _ in det_array], dtype=cp.int32)

                max_r = int(cp.max(rads).get())

                idx_x = cp.arange(-max_r, max_r + 1)
                idx_y = cp.arange(-max_r, max_r + 1)

                grid_x = cx[:, None, None] + idx_x[None, None, :]
                grid_y = cy[:, None, None] + idx_y[None, :, None]

                oob_mask = (grid_x < 0) | (grid_x >= W) | (grid_y < 0) | (grid_y >= H)

                grid_x = cp.clip(grid_x, 0, W-1)
                grid_y = cp.clip(grid_y, 0, H-1)

                patches_lab = lab_gpu[grid_y, grid_x]
                patches_hsv = hsv_gpu[grid_y, grid_x]

                Y_grid, X_grid = cp.meshgrid(idx_y, idx_x, indexing='ij')
                dist_sq = X_grid[None, :, :]**2 + Y_grid[None, :, :]**2

                circle_mask = dist_sq <= (rads[:, None, None] ** 2)
                valid_mask = circle_mask & (~oob_mask)

                L_vals = patches_lab[..., 0]
                S_vals = patches_hsv[..., 1]
                non_glare = (L_vals < glare_threshold) | (S_vals > 30)

                non_glare_valid = valid_mask & non_glare
                has_non_glare = cp.sum(non_glare_valid, axis=(1,2)) > 0

                final_mask = cp.where(has_non_glare[:, None, None], non_glare_valid, valid_mask)

                S_masked = S_vals * final_mask
                sum_S = cp.sum(S_masked, axis=(1,2))
                has_S = sum_S > 0

                sum_mask = cp.sum(final_mask, axis=(1,2))
                sum_mask_safe = cp.maximum(sum_mask, 1)

                weights = cp.where(
                    has_S[:, None, None],
                    S_masked / cp.maximum(sum_S[:, None, None], 1e-5),
                    final_mask.astype(cp.float32) / sum_mask_safe[:, None, None]
                )

                mean_lab = cp.sum(patches_lab * weights[..., None], axis=(1,2))
                mean_S = cp.sum(S_vals * final_mask, axis=(1,2)) / sum_mask_safe

                return mean_lab, mean_S

            if track_lab_S is not None:
                track_lab_gpu = to_gpu(np.asarray(track_lab_S[0], dtype=np.float32))
                track_S_gpu = to_gpu(np.asarray(track_lab_S[1], dtype=np.float32))
            else:
                track_lab_gpu, track_S_gpu = get_weighted_color_batch_gpu(track_dets)
            det_lab_gpu, det_S_gpu = get_weighted_color_batch_gpu(det_list)

            # Vectorized distance computation on GPU
            # Compute pairwise distances: ||track_lab[:, None, :] - det_lab[None, :, :]||
            diff = track_lab_gpu[:, None, :] - det_lab_gpu[None, :, :]  # (M, N, 3)
            dist_matrix_gpu = cp.linalg.norm(diff, axis=2)  # (M, N)
            dist_matrix = to_numpy(dist_matrix_gpu)

            # Saturation confidence
            S_conf_gpu = cp.minimum(track_S_gpu[:, None], det_S_gpu[None, :]) / 255.0  # (M, N)

            if saturation_weight:
                S_conf_gpu = cp.clip((S_conf_gpu - 0.1) / 0.4, 0.0, 1.0)

            confidence_matrix = to_numpy(S_conf_gpu)

            # Non-linear penalty
            if nonlinear_k > 0:
                dist_matrix = (np.exp(nonlinear_k * dist_matrix / 100.0) - 1.0) * (100.0 / nonlinear_k)

            return dist_matrix.astype(np.float32), confidence_matrix.astype(np.float32)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ColorGPU] Vectorization failed: {e}. Falling back to CPU loop.")

    # CPU fallback
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    H, W = frame_bgr.shape[:2]

    def get_weighted_color_cpu(det):
        """Extract saturation-weighted LAB color for one detection (CPU path)."""
        x, y, r, _ = det
        cx, cy = int(round(x)), int(round(y))
        rad = int(max(2, round(r*1.15)))

        x0 = max(0, cx - rad)
        y0 = max(0, cy - rad)
        x1 = min(W, cx + rad + 1)
        y1 = min(H, cy + rad + 1)

        if x1 <= x0 or y1 <= y0:
            return np.array([0., 0., 0.]), 0.0

        patch_lab = lab[y0:y1, x0:x1]
        patch_hsv = hsv[y0:y1, x0:x1]

        # Localized circular mask
        Y, X = np.ogrid[:y1-y0, :x1-x0]
        dist_sq = (X - (cx - x0))**2 + (Y - (cy - y0))**2
        mask = dist_sq <= rad**2

        lab_vals = patch_lab[mask]
        hsv_vals = patch_hsv[mask]

        if len(lab_vals) == 0:
            return np.array([0., 0., 0.]), 0.0

        L_vals = lab_vals[:, 0]
        S_vals = hsv_vals[:, 1]
        non_glare = (L_vals < glare_threshold) | (S_vals > 30)

        if np.sum(non_glare) > 0:
            lab_filtered = lab_vals[non_glare]
            S_filtered = S_vals[non_glare]
        else:
            lab_filtered = lab_vals
            S_filtered = S_vals

        if len(S_filtered) > 0 and S_filtered.sum() > 0:
            S_weights = S_filtered / S_filtered.sum()
            mean_lab = np.average(lab_filtered, axis=0, weights=S_weights)
        else:
            mean_lab = lab_filtered.mean(axis=0)

        mean_S = float(S_filtered.mean()) if len(S_filtered) > 0 else 0.0

        return mean_lab, mean_S

    # Pre-compute all colors to avoid M*N re-computations
    if track_lab_S is not None:
        track_colors_cpu = [(np.asarray(track_lab_S[0][i], dtype=np.float64), float(track_lab_S[1][i]))
                            for i in range(M)]
    else:
        track_colors_cpu = [get_weighted_color_cpu(t) for t in track_dets]
    det_colors_cpu = [get_weighted_color_cpu(d) for d in det_list]

    return _pair_matrices_cpu(track_colors_cpu, det_colors_cpu, saturation_weight, nonlinear_k)


def _pair_matrices_scalar(track_colors_cpu, det_colors_cpu, saturation_weight, nonlinear_k, pairs=None):
    """The original per-pair loop (reference semantics). ``pairs``: optional (i, j) list to restrict it to."""
    M, N = len(track_colors_cpu), len(det_colors_cpu)
    dist_matrix = np.zeros((M, N), np.float32)
    confidence_matrix = np.zeros((M, N), np.float32)
    for i, j in (pairs if pairs is not None else ((i, j) for i in range(M) for j in range(N))):
        track_lab_val, track_S_val = track_colors_cpu[i]
        det_lab_val, det_S_val = det_colors_cpu[j]

        delta = np.linalg.norm(track_lab_val - det_lab_val)
        S_conf = min(track_S_val, det_S_val) / 255.0

        if saturation_weight:
            S_conf = np.clip((S_conf - 0.1) / 0.4, 0.0, 1.0)

        if nonlinear_k > 0:
            delta = (np.exp(nonlinear_k * delta / 100.0) - 1.0) * (100.0 / nonlinear_k)

        dist_matrix[i, j] = delta
        confidence_matrix[i, j] = S_conf
    return dist_matrix, confidence_matrix


_SQNORM_BY_MATMUL = None


def _sqnorm_rows(d):
    """x.dot(x) for every row of d (K, 3), bit-identical to np.linalg.norm's 1-D path (sqrt(x.dot(x))).

    Batched (K,1,3) @ (K,3,1) matmul hits the same BLAS dot kernel; a one-time self-test confirms that on this
    platform (else: exact per-row .dot loop)."""
    global _SQNORM_BY_MATMUL
    if _SQNORM_BY_MATMUL is None:
        rng = np.random.default_rng(12345)
        ok = True
        for dt in (np.float64, np.float32):
            t = rng.uniform(-255.0, 255.0, (4096, 3)).astype(dt)
            ref = np.array([r.dot(r) for r in t], dtype=dt)
            ok &= bool(np.array_equal((t[:, None, :] @ t[:, :, None])[:, 0, 0], ref))
        _SQNORM_BY_MATMUL = ok
    if _SQNORM_BY_MATMUL:
        return (d[:, None, :] @ d[:, :, None])[:, 0, 0]
    return np.array([r.dot(r) for r in d], dtype=d.dtype)


def _pair_matrices_cpu(track_colors_cpu, det_colors_cpu, saturation_weight, nonlinear_k):
    """Vectorised _pair_matrices_scalar, bit-identical to it.

    The scalar code works in each pair's promoted dtype (np.result_type of the two LAB arrays: float32 when both
    come from the float32 weighted mean, float64 otherwise), so the matrix is filled per (track dtype, det dtype)
    block in that dtype. Each step matches its scalar counterpart: elementwise subtraction, sqrt(x.dot(x)) via
    _sqnorm_rows, elementwise min / divide / clip / exp (numpy >= 2 promotion: python float * float32 stays
    float32, as for the scalar), float32 cast on store. Other dtypes take the scalar code.
    """
    M, N = len(track_colors_cpu), len(det_colors_cpu)
    t_lab = [np.asarray(c[0]) for c in track_colors_cpu]
    d_lab = [np.asarray(c[0]) for c in det_colors_cpu]
    tS = np.array([c[1] for c in track_colors_cpu], dtype=np.float64)
    dS = np.array([c[1] for c in det_colors_cpu], dtype=np.float64)

    S_conf = np.minimum(tS[:, None], dS[None, :]) / 255.0
    if saturation_weight:
        S_conf = np.clip((S_conf - 0.1) / 0.4, 0.0, 1.0)
    confidence_matrix = S_conf.astype(np.float32)
    dist_matrix = np.zeros((M, N), np.float32)

    floats = (np.dtype(np.float32), np.dtype(np.float64))
    t_dt = [x.dtype for x in t_lab]
    d_dt = [x.dtype for x in d_lab]
    for tdt in set(t_dt):
        ti = [i for i in range(M) if t_dt[i] == tdt]
        for ddt in set(d_dt):
            dj = [j for j in range(N) if d_dt[j] == ddt]
            if tdt not in floats or ddt not in floats:
                pairs = [(i, j) for i in ti for j in dj]
                d_ref, _ = _pair_matrices_scalar(track_colors_cpu, det_colors_cpu, saturation_weight, nonlinear_k,
                                                 pairs)
                dist_matrix[np.ix_(ti, dj)] = d_ref[np.ix_(ti, dj)]
                continue
            rt = np.result_type(tdt, ddt)
            T = np.array([t_lab[i] for i in ti], dtype=rt).reshape(len(ti), 3)
            D = np.array([d_lab[j] for j in dj], dtype=rt).reshape(len(dj), 3)
            diff = (T[:, None, :] - D[None, :, :]).reshape(len(ti) * len(dj), 3)
            delta = np.sqrt(_sqnorm_rows(diff)).reshape(len(ti), len(dj))
            if nonlinear_k > 0:
                delta = (np.exp(nonlinear_k * delta / 100.0) - 1.0) * (100.0 / nonlinear_k)
            dist_matrix[np.ix_(ti, dj)] = delta
    return dist_matrix, confidence_matrix

def vectorized_distance_matrix_gpu(track_positions, det_positions, use_gpu=True):
    """
    Compute pairwise distance matrix between tracks and detections

    Args:
        track_positions: Array (M, 2) of [x, y] positions
        det_positions: Array (N, 2) of [x, y] positions
        use_gpu: Whether to use GPU

    Returns:
        distance_matrix: (M, N) array of Euclidean distances
    """
    M = track_positions.shape[0]
    N = det_positions.shape[0]

    if M == 0 or N == 0:
        return np.zeros((M, N), np.float32)

    # Try GPU path
    if use_gpu and gpu.USE_CUPY:
        try:
            track_gpu = to_gpu(track_positions.astype(np.float32))  # (M, 2)
            det_gpu = to_gpu(det_positions.astype(np.float32))  # (N, 2)

            # Vectorized distance: ||track[:, None, :] - det[None, :, :]||
            diff = track_gpu[:, None, :] - det_gpu[None, :, :]  # (M, N, 2)
            dist_matrix_gpu = cp.linalg.norm(diff, axis=2)  # (M, N)

            return to_numpy(dist_matrix_gpu).astype(np.float32)
        except Exception as e:
            print(f"⚠️  GPU distance matrix failed: {e}, falling back to CPU")

    # CPU fallback
    # Vectorized numpy computation
    diff = track_positions[:, None, :] - det_positions[None, :, :]  # (M, N, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)  # (M, N)

    return dist_matrix.astype(np.float32)

