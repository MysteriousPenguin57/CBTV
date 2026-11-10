"""Kernel-recovery pass for unmatched tracks (occlusion/overlap)."""
import math

import cv2
import numpy as np

from .assignment import compute_kernel_confidence
from .correlation import ncc
from .kernels import build_LAB_kernel_cpu_optimized, compute_particle_colours_lab_from_split


def batch_kernel_recovery(unmatched_tracks, tracks, previous_detections, previous_frame,
                         current_frame, args, current_detections=None):
    """
    Batch kernel recovery for unmatched tracks (GPU-accelerated)

    KEY OPTIMIZATIONS:
    - Convert current frame LAB once (not N times)
    - Convert previous frame LAB once (not N times)
    - Build all kernels in batch
    - Correlate all patches in parallel

    Returns:
        List of (track_idx, det_x, det_y, det_r, peak_val) for successful recoveries
    """
    if not unmatched_tracks or previous_frame is None or previous_detections is None:
        return []

    # Convert frames to LAB ONCE for all operations
    prev_lab = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2LAB)
    prev_L, prev_A, prev_B = cv2.split(prev_lab)
    prev_H, prev_W = prev_L.shape

    curr_lab = cv2.cvtColor(current_frame, cv2.COLOR_BGR2LAB)

    # Pre-compute colors for previous detections ONCE
    col_lab_prev = compute_particle_colours_lab_from_split(prev_L, prev_A, prev_B, previous_detections)

    # Prepare batch data
    recovery_tasks = []

    # [REC-1] Gather unmatched tracks + neighbour proximity filter
    for track_idx in unmatched_tracks:
        t = tracks[track_idx]

        if t.missed >= 2:
            continue

        # Find neighbors (vectorized)
        neigh = []
        for pi, (px, py, pr, _) in enumerate(previous_detections):
            if math.hypot(t.x - px, t.y - py) <= args.neighbor_search_factor * max(t.r, 1.0):
                neigh.append(pi)

        if len(neigh) < args.kernel_recovery_min_neighbors:
            continue

        # Find track's previous position
        track_prev_idx = None
        min_dist = float('inf')
        for pi, (px, py, pr, _) in enumerate(previous_detections):
            pd = math.hypot(t.x - px, t.y - py)
            if pd < min_dist:
                min_dist = pd
                track_prev_idx = pi

        if track_prev_idx is None:
            continue

        recovery_tasks.append({
            'track_idx': track_idx,
            'track': t,
            'track_prev_idx': track_prev_idx,
            'neighbors': neigh,
        })

    if not recovery_tasks:
        return []

    # Phase 3l fix: build a single np.ndarray of fresh current-frame
    # detection positions. Each task gets a reference (no copy) so the
    # snap-to-fresh-detection logic in the inner loop can find the
    # nearest fresh detection for the NCC peak.
    if current_detections:
        _curr_snap_pos = np.array([[d[0], d[1]] for d in current_detections],
                                  dtype=np.float32)
    else:
        _curr_snap_pos = np.empty((0, 2), dtype=np.float32)
    for _t in recovery_tasks:
        _t['all_dets_snap'] = _curr_snap_pos

    # Batch build all kernels using shared LAB data
    kernels_data = []
    for task in recovery_tasks:
        try:
            K, cx_off, cy_off = build_LAB_kernel_cpu_optimized(
                previous_detections, task['track_prev_idx'], task['neighbors'],
                prev_L, prev_A, prev_B, col_lab_prev, prev_H, prev_W,
                task['track'].r, args
            )
            kernels_data.append((task, K, cx_off, cy_off))
        except Exception:
            continue

    if not kernels_data:
        return []

    # [REC-2] ROI extraction + search radius setup
    # Batch correlate all patches
    recoveries = []
    sr = int(args.kernel_recovery_search_radius)

    for task, K, cx_off, cy_off in kernels_data:
        try:
            t = task['track']
            kh, kw = K.shape[:2]

            pred_x, pred_y = t.x, t.y

            x0 = max(0, int(pred_x) - (kw//2 + sr))
            x1 = min(current_frame.shape[1], int(pred_x) + (kw//2 + sr) + 1)
            y0 = max(0, int(pred_y) - (kh//2 + sr))
            y1 = min(current_frame.shape[0], int(pred_y) + (kh//2 + sr) + 1)

            # Extract patch from current LAB frame (already converted)
            patch_lab = curr_lab[y0:y1, x0:x1]

            if patch_lab.shape[0] >= kh and patch_lab.shape[1] >= kw:
                # Correlate using LAB patch (no conversion needed!)
                Lp, Ap, Bp = cv2.split(patch_lab.astype(np.float32))
                Lk, Ak, Bk = cv2.split(K.astype(np.float32))
                # [REC-3] NCC correlation and confidence gate
                RL = ncc(Lp, Lk)
                RA = ncc(Ap, Ak)
                RB = ncc(Bp, Bk)
                R = (RL + RA + RB) / 3.0

                peak_val = float(R.max())

                # Use relative scoring instead of hard threshold
                # Check if peak is clearly above background (uniqueness + SNR)
                confidence = compute_kernel_confidence(R, peak_val)

                if confidence >= args.kernel_recovery_confidence:
                    peak_y, peak_x = np.unravel_index(R.argmax(), R.shape)

                    ncc_x = float(x0 + peak_x + kw//2)
                    ncc_y = float(y0 + peak_y + kh//2)

                    # [Phase 3l fix 2026-04-30] Snap NCC peak to the nearest
                    # FRESH detection within (t.r * 1.5) px. Original code
                    # wrote the NCC peak directly as the matched position,
                    # which biased output positions toward the cached
                    # kernel template's anchor (a previous-frame
                    # detection). At step>1 that bias becomes ~3 px median
                    # and busts a 6 px tolerance. The tracking kernel is a
                    # SCORING tool; positions must come from fresh
                    # detections.
                    snap_radius = max(t.r * 1.5, 12.0)
                    snap_pos = task.get('all_dets_snap', None)
                    det_x = ncc_x
                    det_y = ncc_y
                    snapped = False
                    if snap_pos is not None and len(snap_pos) > 0:
                        dx = snap_pos[:, 0] - ncc_x
                        dy = snap_pos[:, 1] - ncc_y
                        dists_local = np.sqrt(dx * dx + dy * dy)
                        di = int(np.argmin(dists_local))
                        if dists_local[di] <= snap_radius:
                            det_x = float(snap_pos[di, 0])
                            det_y = float(snap_pos[di, 1])
                            snapped = True
                    if not snapped:
                        # No fresh detection nearby - skip (track stays
                        # unmatched, will likely be terminated soon).
                        continue
                    det_r = t.r

                    # [REC-4] Per-track position update + virtual detection emit
                    dist_moved = math.hypot(det_x - t.x, det_y - t.y)
                    if dist_moved <= args.kernel_recovery_max_move:
                        recoveries.append((task['track_idx'], det_x, det_y, det_r, peak_val))
        except Exception:
            continue

    return recoveries

