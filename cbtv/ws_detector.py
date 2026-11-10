"""FP-resistant watershed detector adapter for the cbtv linker.

Purpose (Phase 1 noisy-comparison fairness)
-------------------------------------------
On noisy dense video (K01_realistic_balanced) the original Hough/LoG
detector FP-explodes — so a cbtv-vs-GOTH comparison there measures the
*detector*, not the *linker*.  GOTH already sidesteps this by detecting
with ``ws_multi_vote_gpu`` (a watershed multi-channel voting detector that
is FP-resistant) and only linking with its GNN.

This module lets the cbtv linker consume the **same** ``ws_multi_vote_gpu``
detections — the identical strategy function the Phase-3 benchmarks,
tracker_A and tracker_GOTH all use (see ``trackers/_detection_swap.py``).
With detection held fixed and FP-resistant, the only moving part is the
linker → we isolate the tracker, exactly the CBTV-U / tracker-A pattern
from the older tests.

It is wired as the ``ws_multi_vote`` preset in :mod:`cbtv.detectors`;
the default ``bayes_adapt`` path is untouched (byte-identical).
"""
from __future__ import annotations

_DB = None
_STRATEGY_FN = None


def _ensure_loaded(strategy_name: str):
    """Lazy-import cbtv.detect (ex detection_benchmark) + pin the Phase-3b/3d benchmark flags.

    These module-level flags are read inside build_channels / the strategy;
    pinning them once reproduces the benchmark's F1 (watershed-peaks ON,
    median-kernel ON, bg-subtract ON, hough-multi ON, clip/fixed-bg OFF).
    """
    global _DB, _STRATEGY_FN
    if _DB is None:
        from .detect import channels as _db
        # [DET-SWAP-2] Pin runtime flags == Phase-3b/3d benchmark config.
        _db.USE_WATERSHED_PEAKS = True
        _db.USE_BORDER_PAD = False
        _db.USE_MEDIAN_KERNEL = True
        _db.USE_OBSTACLE_MASK = False
        _db.SUBTRACT_BACKGROUND = True
        _db.TOPHAT_THRESHOLD = False
        _db.HOUGH_MULTI_CHANNEL = True
        _db.USE_FIXED_BG = False
        _db.CONTINUOUS_VOTING = False
        _db.GAUSSIAN_CENTROID = False
        if hasattr(_db, "CLIP_NOISE_FLOOR"):
            _db.CLIP_NOISE_FLOOR = False
        if hasattr(_db, "TEMPORAL_KERNEL_ACTIVE"):
            _db.TEMPORAL_KERNEL_ACTIVE = False
        _DB = _db
    if _STRATEGY_FN is None:
        from .detect import strategies as _scb
        STRATEGIES = _scb.STRATEGIES
        if strategy_name not in STRATEGIES:
            raise ValueError(f"unknown ws strategy {strategy_name!r}; "
                             f"known: {sorted(STRATEGIES)}")
        # Without gpu_strategies the *_gpu strategies return zero detections on
        # every frame; fail here instead.
        if (strategy_name.endswith("_gpu")
                and not getattr(_scb, "_GPU_STRATEGIES_AVAILABLE", True)):
            raise ImportError(
                f"detector {strategy_name!r} needs cbtv/detect/gpu_strategies.py and "
                f"cbtv/detect/gpu_backend.py, but importing them "
                f"failed ({getattr(_scb, '_GPU_STRATEGIES_IMPORT_ERROR', None)}). "
                f"Reinstall CBTV or use --detector csv/cellpose.")
        # ws_vgl_log_gpu has no CPU path (its Laplacian is a CuPy RawKernel): without one it returned 0 detections
        # on every frame, silently. Fail here; ws_multi_vote_gpu does fall back to the CPU.
        if strategy_name == "ws_vgl_log_gpu":
            from .detect import gpu_backend as _gb
            if not (_gb.HAS_CUPY and _gb.HAS_CUPY_RAWKERNEL):
                raise RuntimeError(
                    "detector 'ws_vgl_log' runs on the GPU only (needs CuPy with RawKernel support and a CUDA "
                    "device) and has no CPU path; on CPU use --detector ws_multi_vote (same watershed family), "
                    "csv or bayes_adapt.")
        _STRATEGY_FN = STRATEGIES[strategy_name]
    return _DB, _STRATEGY_FN


def detect_ws(frame_bgr, r_min, r_max, strategy_name="ws_multi_vote_gpu"):
    """Run the watershed strategy, return cbtv's detection 5-tuple.

    Mirrors ``user_detection_with_bayes_adapt``'s return contract:
        (dets_raw, dets_bayes_added, gray, chans, dbg)
    where dets_raw is a list of (x, y, r, score).  dets_bayes_added is
    always [] (watershed has no Bayesian post-step), which the downstream
    filter/cost-matrix code handles (it just concatenates the two lists).
    """
    db, strategy_fn = _ensure_loaded(strategy_name)

    # [DET-SWAP-3] Background-subtract then build the benchmark channel set,
    # so the strategy sees the exact inputs it was tuned on.
    bgr_input = frame_bgr
    if getattr(db, "SUBTRACT_BACKGROUND", False):
        try:
            bgr_input = db.subtract_background(frame_bgr)
        except Exception:
            bgr_input = frame_bgr
    chans, gray = db.build_channels(bgr_input)

    # [DET-SWAP-4] Run the FP-resistant detector.
    try:
        detections, elapsed, diag, _surface = strategy_fn(
            bgr_input, chans, gray, int(r_min), int(r_max))
    except Exception as e:
        print(f"[DET-SWAP] {strategy_name} raised: {e}")
        detections, elapsed, diag = [], 0.0, {"error": str(e)}

    # [DET-SWAP-5] Normalise to (x, y, r, score) 4-tuples.
    norm = []
    for d in detections:
        if len(d) >= 4:
            norm.append((float(d[0]), float(d[1]), float(d[2]), float(d[3])))
        elif len(d) == 3:
            norm.append((float(d[0]), float(d[1]), float(d[2]), 0.5))

    dbg = {"method": strategy_name, "raw": len(norm), "bayes_added": 0,
           "elapsed_ms": float(elapsed) * 1000.0, "is_log_fallback": False}
    if isinstance(diag, dict):
        for k, v in diag.items():
            if isinstance(v, (int, float, str, bool)):
                dbg[f"strat_{k}"] = v
    return norm, [], gray, chans, dbg
