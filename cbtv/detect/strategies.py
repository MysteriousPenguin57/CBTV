"""Pruned from the research module ``strategy_combination_benchmark.py``: only the code reachable from the shipped
detectors (csv, ws_multi_vote / ws_multi_vote_gpu, ws_vgl_log, cellpose) is kept, unchanged.

Original module docstring:

strategy_combination_benchmark.py

Comprehensive detection-strategy benchmark: ~20 combinations of watershed,
NCC kernel, Hough, and filtering approaches across 10 videos in the CBTV
particle-tracking dataset.

This module ONLY consumes public APIs from `detection_benchmark.py` and
`detection_diagnostics.py`. It does not modify those files.
"""

import numpy as np

# The Phase 2D module flags live on the detection_benchmark (now cbtv.detect.channels) module itself
# so helpers (ncc_correlate, build_kernel_from_exemplars, extract_peaks_dispatch,
# filter_detections_by_mask) all see the same toggles. Expose under short name.
from . import channels as _db

np.random.seed(42)

def _merge_votes(det_lists, tol, min_consensus=None, score_weight_consensus=None):
    """Greedy multi-source vote merge with optional continuous consensus weighting.

    det_lists: list of per-source detection lists, each element (x, y, r, score).
    A detection cluster is KEPT when its consensus_factor (= unique source count
    / total source count) is >= ``min_consensus``. Output score is multiplied
    by consensus_factor when ``score_weight_consensus`` is True.

    Phase 2J defaults: when both args are None we look up the runtime values
    on the detection_benchmark module (CONTINUOUS_VOTING flag). Off => binary
    2-of-3 with min_consensus = 2/3 (back-compat with prior behaviour, where
    n_sources = 3 and the original test was len(srcs) >= 2). On =>
    min_consensus = 0.34 (1-of-N) and score scaled by consensus_factor.
    """
    if min_consensus is None or score_weight_consensus is None:
        if _db.CONTINUOUS_VOTING:
            min_consensus = 0.34 if min_consensus is None else min_consensus
            score_weight_consensus = (True if score_weight_consensus is None
                                      else score_weight_consensus)
        else:
            # Original 2-of-3 behaviour for n_sources == 3.
            min_consensus = 2.0 / 3.0 if min_consensus is None else min_consensus
            score_weight_consensus = (False if score_weight_consensus is None
                                      else score_weight_consensus)
    n_sources = len(det_lists)
    all_dets = []
    for src_idx, lst in enumerate(det_lists):
        for item in lst:
            x, y, r, s = item[0], item[1], item[2], item[3]
            all_dets.append((float(x), float(y), float(r), float(s), src_idx))
    if not all_dets:
        return []
    claimed = [False] * len(all_dets)
    out = []
    for i in range(len(all_dets)):
        if claimed[i]:
            continue
        x, y, r, s, src = all_dets[i]
        members = [i]
        for j in range(i + 1, len(all_dets)):
            if claimed[j]:
                continue
            xj, yj, rj, sj, srcj = all_dets[j]
            if (x - xj) ** 2 + (y - yj) ** 2 <= tol * tol:
                members.append(j)
        srcs = set(all_dets[m][4] for m in members)
        consensus_factor = len(srcs) / float(max(n_sources, 1))
        if consensus_factor + 1e-9 >= min_consensus:
            xs = float(np.mean([all_dets[m][0] for m in members]))
            ys = float(np.mean([all_dets[m][1] for m in members]))
            rs = float(np.mean([all_dets[m][2] for m in members]))
            ss_raw = float(np.mean([all_dets[m][3] for m in members]))
            ss = ss_raw * consensus_factor if score_weight_consensus else ss_raw
            out.append((xs, ys, rs, ss))
        for m in members:
            claimed[m] = True
    return out

def _empty_result(method, elapsed, note=None):
    diag = {'method': method, 'empty': True}
    if note is not None:
        diag['note'] = note
    return ([], elapsed, diag, None)

try:
    from .gpu_strategies import (
        ws_multi_vote_gpu as _ws_multi_vote_gpu_impl,
    )
    from .gpu_strategies import (
        ws_vgl_log_gpu as _ws_vgl_log_gpu_impl,
    )
    _GPU_STRATEGIES_AVAILABLE = True
    _GPU_STRATEGIES_IMPORT_ERROR = None
except Exception as _e:
    _GPU_STRATEGIES_AVAILABLE = False
    _GPU_STRATEGIES_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    _ws_multi_vote_gpu_impl = None  # type: ignore[assignment]
    _ws_vgl_log_gpu_impl = None  # type: ignore[assignment]

def ws_multi_vote_gpu(bgr, channels, gray, rmin, rmax, topk=3):
    if not _GPU_STRATEGIES_AVAILABLE or _ws_multi_vote_gpu_impl is None:
        return _empty_result('ws_multi_vote_gpu', 0.0, note='gpu_unavailable')
    # Inject our _merge_votes (the strategy module's version honours
    # CONTINUOUS_VOTING module flag).
    return _ws_multi_vote_gpu_impl(bgr, channels, gray, rmin, rmax, topk,
                                    _merge_votes=_merge_votes)

def ws_vgl_log_gpu(bgr, channels, gray, rmin, rmax, topk=3):
    if not _GPU_STRATEGIES_AVAILABLE or _ws_vgl_log_gpu_impl is None:
        return _empty_result('ws_vgl_log_gpu', 0.0, note='gpu_unavailable')
    return _ws_vgl_log_gpu_impl(bgr, channels, gray, rmin, rmax, topk)


# Only the shipped strategies (the ws_multi_vote / ws_vgl_log presets in cbtv/detectors.py); the research table
# in strategy_combination_benchmark.py had 29 entries.
STRATEGIES = {
    'ws_multi_vote_gpu': ws_multi_vote_gpu,
    'ws_vgl_log_gpu': ws_vgl_log_gpu,
}
