"""LDA features + temporal Bayesian detection refinement."""
import math

import cv2
import numpy as np

from . import gpu
from .correlation import multiscale_corr
from .detection_filter import peaks_from_map
from .exemplars import norm_kernel_from_exemplars
from .gpu import cp, to_gpu


def extract_lda_features(chans, gray, x, y, r, correlation_maps, dets_list=None):
    """
    Extract discriminative feature vector for LDA classification.

    Args:
        chans: dict of RGB channels
        gray: grayscale image
        x, y, r: candidate location and radius
        correlation_maps: dict of correlation maps at multiple scales
        dets_list: list of existing detections for context features

    Returns:
        numpy array of features (19-dimensional)
    """
    features = []
    H, W = gray.shape

    xi, yi = int(round(x)), int(round(y))
    ri = int(round(r))

    # 1. Multi-channel correlations (9 features: 3 scales × 3 channels)
    for scale_key in ['0.9', '1.0', '1.1']:
        for chan_name in ['R', 'G', 'B']:
            if correlation_maps and chan_name in correlation_maps and scale_key in correlation_maps[chan_name]:
                corr_map = correlation_maps[chan_name][scale_key]
                if 0 <= yi < corr_map.shape[0] and 0 <= xi < corr_map.shape[1]:
                    features.append(float(corr_map[yi, xi]))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

    # 2. Local statistics (4 features)
    y0, y1 = max(0, yi - ri), min(H, yi + ri)
    x0, x1 = max(0, xi - ri), min(W, xi + ri)

    if y1 > y0 and x1 > x0:
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        features.append(float(np.mean(patch)))       # Intensity
        features.append(float(np.std(patch)))        # Contrast
        features.append(float(np.min(patch)))        # Dark background
        features.append(float(np.max(patch)))        # Bright particle
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # 3. Gradient features (2 features)
    if y1 > y0 and x1 > x0:
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features.append(float(np.mean(grad_mag)))    # Edge strength

        # Circularity: std of gradient direction (low = circular)
        grad_dir = np.arctan2(gy, gx)
        features.append(float(np.std(grad_dir)))     # Circularity (lower = more circular)
    else:
        features.extend([0.0, 0.0])

    # 4. Symmetry features (2 features)
    if y1 > y0 and x1 > x0 and patch.shape[0] > 4 and patch.shape[1] > 4:
        # Radial symmetry: compare opposite quadrants
        h, w = patch.shape
        hh, hw = h // 2, w // 2
        if hh > 0 and hw > 0:
            q1 = patch[:hh, :hw]
            q3 = patch[hh:, hw:]
            q3_flip = np.flip(q3)

            min_h = min(q1.shape[0], q3_flip.shape[0])
            min_w = min(q1.shape[1], q3_flip.shape[1])

            if min_h > 0 and min_w > 0:
                radial_sym = float(np.corrcoef(
                    q1[:min_h, :min_w].flatten(),
                    q3_flip[:min_h, :min_w].flatten()
                )[0, 1])
                features.append(radial_sym if not np.isnan(radial_sym) else 0.0)
            else:
                features.append(0.0)

            # Horizontal symmetry
            left = patch[:, :hw]
            right = np.fliplr(patch[:, hw:])
            min_w2 = min(left.shape[1], right.shape[1])
            if min_w2 > 0:
                horiz_sym = float(np.corrcoef(
                    left[:, :min_w2].flatten(),
                    right[:, :min_w2].flatten()
                )[0, 1])
                features.append(horiz_sym if not np.isnan(horiz_sym) else 0.0)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    # 5. Context features (2 features)
    if dets_list and len(dets_list) > 0:
        # Local density: count nearby detections
        nearby_count = sum(1 for dx, dy, *_ in dets_list
                          if 0 < math.hypot(x - dx, y - dy) < 3.0 * r)
        features.append(float(nearby_count))

        # Distance to nearest detection
        min_dist = min([math.hypot(x - dx, y - dy) for dx, dy, *_ in dets_list],
                      default=float('inf'))
        features.append(float(min_dist / r))  # Normalized by radius
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)

def train_lda_discriminant(positives, negatives, chans, gray, correlation_maps):
    """
    Train LDA classifier to discriminate particles from background.

    Args:
        positives: list of high-confidence detections
        negatives: list of synthetic negative examples
        chans: dict of RGB channels
        gray: grayscale image
        correlation_maps: precomputed correlation maps

    Returns:
        Trained LinearDiscriminantAnalysis model, or None if training fails
    """
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError:
        return None

    # Extract features for positives
    X_pos = []
    for x, y, r, *rest in positives[:300]:  # Limit to top 300
        feats = extract_lda_features(chans, gray, x, y, r, correlation_maps, positives)
        X_pos.append(feats)

    # Extract features for negatives (balance classes)
    X_neg = []
    for x, y, r in negatives[:len(X_pos)]:
        feats = extract_lda_features(chans, gray, x, y, r, correlation_maps, positives)
        X_neg.append(feats)

    if len(X_pos) < 5 or len(X_neg) < 5:
        return None

    # Combine and label
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))

    # Check for NaN/Inf
    if np.any(~np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    # Train LDA
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        return lda
    except Exception:
        return None

def improved_bayesian_refinement_temporal(
    chans, gray, fused, dets_raw, rad_use, names, weights, results,
    fp_memory, refine_top_pos=300, refine_beta=0.6, scales=(0.9,1.0,1.1),
    thr_abs=0.4, thr_mode="hybrid", lk=2.5, lksz=0, thr_topk_tile=0, thr_tile_factor=3.0, nms_rad=None
):
    # [Phase 4D] Temporal Bayesian refinement with false-positive memory.
    # Pipeline diagram: see docs/architecture/PIPELINE_FLOWCHARTS.md
    # Flowchart 17 (Bayesian Refinement with FP Memory). Section labels
    # [BAY-1..5] map 1:1 to that diagram's stages.
    #
    # [v6 Change] Original pipeline trained a single static LDA discriminant
    # on synthetic negatives generated from frame 0 and applied it unchanged
    # to all 401 frames. Phase 4D added the FalsePositiveMemory ring buffer
    # (10 frames, 50 negatives/frame) and the temporal Bayes update so that
    # recurring spurious detections at the same coordinates are suppressed
    # automatically. Result: sustained false-positive rate dropped from ~3 %
    # to <0.5 % on the primary video.
    #
    # [Decision Rationale] fp_memory_decay (max_history_frames=10) was chosen
    # to balance two failure modes: too short (≤3 frames) lets transient noise
    # leak through; too long (≥30 frames) carries stale FPs into frames where
    # the underlying scene has changed. 10 frames is roughly the lifetime of
    # a transient artefact in this dataset.
    """
    Improved Bayesian refinement using temporal history of false positives
    """
    if len(dets_raw) < 8:
        return []

    H, W = gray.shape

    # === [BAY-1] LDA feature / positive prototype extraction (Flowchart 17 — Likelihood) ===
    # Get positive examples (high-scoring current detections)
    pos = sorted(dets_raw, key=lambda d: d[3], reverse=True)[:refine_top_pos]

    # === [BAY-3] FP memory lookup (Flowchart 17 — FP Memory Lookup) ===
    # Get negative examples from temporal history
    neg = fp_memory.get_smart_negatives(pos, (H, W), min_separation=2.0 * rad_use)

    # If no history negatives, fall back to original method
    if len(neg) < 5:
        # Use bottom percentile of correlation map (original method)
        if gpu.USE_CUPY:
            fused_gpu = to_gpu(fused)
            corr_min = float(cp.percentile(fused_gpu, 10.0))
        else:
            corr_min = float(np.percentile(fused, 10.0))

        step = max(4, int(0.7*rad_use))
        for y in range(rad_use, H-rad_use, step):
            for x in range(rad_use, W-rad_use, step):
                if fused[y,x] <= corr_min:
                    min_dist = min([math.hypot(x-px, y-py)
                                   for px,py,_,_ in pos],
                                  default=float('inf'))
                    if min_dist > 2.0 * rad_use:
                        neg.append((float(x), float(y), float(rad_use), float(fused[y,x])))
                        if len(neg) >= 50:
                            break
            if len(neg) >= 50:
                break

    # === [BAY-2] Likelihood / kernel build (Flowchart 17 — Bayes Update) ===
    # Build kernels
    if not neg:
        # No negative model, just use positive reinforcement (beta=0)
        adaptive_beta = 0.0
    else:
        # Adaptive beta based on negative quality
        # More negatives from history = more confident = higher beta
        history_neg_ratio = min(len(neg) / 50.0, 1.0)
        adaptive_beta = refine_beta * (0.5 + 0.5 * history_neg_ratio)

    # Build positive and negative kernels
    pos_radii = [r for _,_,r,_ in pos]
    neg_radii = [r for _,_,r,_ in neg] if neg else [rad_use]

    if gpu.USE_CUPY:
        pos_rad = int(cp.median(cp.array(pos_radii))) if pos else rad_use
        neg_rad = int(cp.median(cp.array(neg_radii))) if neg else rad_use
    else:
        pos_rad = int(np.median(pos_radii)) if pos else rad_use
        neg_rad = int(np.median(neg_radii)) if neg else rad_use

    pos_kernels = {}
    neg_kernels = {}

    for n in names:
        pos_kernels[n], _ = norm_kernel_from_exemplars(
            results[n]["chan"],
            [(x,y,pos_rad) for x,y,_,_ in pos],
            pad=0, rad=pos_rad
        )
        if neg:
            neg_kernels[n], _ = norm_kernel_from_exemplars(
                results[n]["chan"],
                [(x,y,neg_rad) for x,y,_,_ in neg],
                pad=0, rad=neg_rad
            )

    # === [BAY-4] Bayes posterior update (Flowchart 17 — Bayes Update) ===
    # bayes_map = max over channels of [w_i × (m_pos - beta × m_neg)]
    # The max-over-channels is the Bayesian "OR" — a candidate is considered
    # particle-like if any single channel has strong positive minus weak
    # negative response. β is adaptive (history_neg_ratio above).
    # Create Bayesian map
    bayes_map = None
    for wi, n in zip(weights, names, strict=False):
        mp = multiscale_corr(results[n]["chan"], pos_kernels[n], scales=scales)

        if neg and n in neg_kernels:
            mn = multiscale_corr(results[n]["chan"], neg_kernels[n], scales=scales)
            m = wi * (mp - adaptive_beta * mn)
        else:
            m = wi * mp

        bayes_map = m if bayes_map is None else np.maximum(bayes_map, m)

    # Normalize and find peaks
    f95 = float(np.percentile(fused, 95.0))
    b95 = float(np.percentile(bayes_map, 95.0))

    scale = (f95+1e-6)/(b95+1e-6) if b95>0 else 1.0

    bayes_norm = np.clip(bayes_map * scale, 0, 1)

    if nms_rad is None:
        nms_rad = rad_use

    dets_bayes = peaks_from_map(bayes_norm, rad_use, thr_abs, nms_rad, mode=thr_mode, local_k=lk, local_ksize=lksz,
                                topk_per_tile=int(thr_topk_tile), tile_factor=float(thr_tile_factor))

    # === [BAY-5] FP memory write-back + temporal decay (Flowchart 17 — Write-back) ===
    # Note: the actual write-back call (fp_memory.add_frame_negatives) is made
    # by the caller after this function returns, so it has access to the full
    # 'killed' list from filter_detections_multistage. The temporal decay is
    # implemented as a FIFO ring of length max_history_frames=10 inside
    # FalsePositiveMemory; older frames fall off automatically.
    # Filter out detections too close to existing
    used = [(x,y) for (x,y,_,_) in dets_raw]
    dets_bayes_added = []
    for (x,y,r,s) in dets_bayes:
        if all(math.hypot(x-ux, y-uy) > max(2.0, 0.5*rad_use) for (ux,uy) in used):
            dets_bayes_added.append((x,y,r,s))

    return dets_bayes_added

