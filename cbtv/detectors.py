"""Detection-strategy registry.

Switch detector with ``--detector NAME``. Each preset is a dict of argument
overrides layered on top of the parsed CLI args, then handed to the one true
detection entry point :func:`user_detection_with_bayes_adapt`. The default
preset ``bayes_adapt`` applies no overrides, so it reproduces the original
call exactly.
"""
from .detection import user_detection_with_bayes_adapt

DETECTOR_PRESETS = {
    "bayes_adapt":  {},                          # full pipeline (default, original behavior)
    "log":          {"force_log": True},         # force LoG exemplar supplement
    "no_bayes":     {"refine_bayes": 0},         # skip Bayesian refinement (Tracker-A style)
    "single_frame": {"multiframe_detection": 0}, # disable multi-frame candidate pooling
    # Phase 1 noisy-comparison fairness: FP-resistant watershed detector
    # (the SAME ws_multi_vote_gpu strategy GOTH / tracker_A use). Routes
    # through cbtv.ws_detector instead of the Hough/LoG path. Lets the
    # cbtv linker consume FP-resistant detections so a cbtv-vs-GOTH
    # comparison on noisy video isolates the LINKER, not the detector.
    "ws_multi_vote": {"_strategy": "ws_multi_vote_gpu"},
    "ws_vgl_log":    {"_strategy": "ws_vgl_log_gpu"},
    # Phase 5 detection-matched Group-M: Cellpose-SAM (CBTV-U). Strong on dense
    # granular flow where Hough/LoG collapses. Routes through cbtv.cellpose_detector.
    "cellpose":      {"_detector": "cellpose"},
    # Phase 6b: shared external CSV (frame,x,y,r,score; proc-frame index) —
    # byte-identical detections to every baseline linker (--detections-csv).
    "csv":           {"_detector": "csv"},
}


def available():
    """Return the sorted list of registered detector names."""
    return sorted(DETECTOR_PRESETS)


def run_detector(name, frame, args, *, thr, dump, frame_buffer_gray,
                 current_frame_idx, fp_memory, obstacle_mask_bool, rng=None):
    """Dispatch to the named detector preset. Returns the detection 5-tuple.

    Args:
        name: preset name, one of :func:`available` (``ws_multi_vote``,
            ``csv``, ``cellpose``, ``bayes_adapt``, ...).
        frame: current BGR frame.
        args: the tracker's parsed options (radius range, detector settings,
            ``--detections-csv`` for ``csv``).
        thr, dump, frame_buffer_gray, current_frame_idx, fp_memory,
        obstacle_mask_bool: per-frame state used by the ``bayes_adapt``
            family; ``current_frame_idx`` (processed index) also selects the
            rows of a ``csv`` detections file.
        rng: ``numpy.random.Generator`` seeded from ``--seed`` for the
            ``bayes_adapt`` exemplar shuffle (None = legacy global state).

    Returns:
        ``(dets, dets_bayes_added, gray, chans, dbg)``: detections as
        ``(x, y, r, score)`` tuples (pixels), extra detections from the
        Bayesian step (empty for the non-bayes presets), the grey frame, the
        channel images used downstream, and a debug dict.
    """
    if name not in DETECTOR_PRESETS:
        raise ValueError(f"unknown detector {name!r}; choices: {available()}")
    ov = DETECTOR_PRESETS[name]
    # Watershed-swap presets route to the FP-resistant detector (same
    # strategy function GOTH / tracker_A use). Detection held fixed ->
    # cbtv-vs-GOTH comparison isolates the linker on noisy video.
    if "_strategy" in ov:
        from .ws_detector import detect_ws
        return detect_ws(frame, args.r_min, args.r_max, strategy_name=ov["_strategy"])
    if ov.get("_detector") == "cellpose":
        from .cellpose_detector import detect_cellpose
        return detect_cellpose(frame, args.r_min, args.r_max)
    if ov.get("_detector") == "csv":
        from .csv_detector import detect_csv
        return detect_csv(frame, args, current_frame_idx)
    def g(k, d):
        return ov[k] if k in ov else getattr(args, k, d)
    return user_detection_with_bayes_adapt(
        frame, args.r_min, args.r_max, thr, topk=args.topk_channels,
        nms_factor=args.nms_factor, hough_polarity=args.hough_polarity,
        refine_bayes=g("refine_bayes", 1), refine_top_pos=args.refine_top_pos,
        refine_top_neg=args.refine_top_neg, refine_beta=args.refine_beta,
        thr_mode=args.thr_mode, thr_local_k=args.thr_local_k, thr_local_ksize=args.thr_local_ksize,
        thr_topk_tile=args.thr_topk_tile, thr_tile_factor=args.thr_tile_factor,
        dedup_threshold=args.dedup_threshold, intelligent_nms=args.intelligent_nms, dump=dump,
        frame_buffer_gray=frame_buffer_gray if g("multiframe_detection", 1) else None,
        current_frame_idx=current_frame_idx,
        multiframe_offset=args.multiframe_offset,
        fp_memory=fp_memory,
        enable_subpixel=bool(args.subpixel_refinement),
        use_antialias=bool(args.antialias_kernels),
        use_simpson=bool(args.better_integration),
        opt_gpu_blob=args.opt_gpu_blob,
        force_log=g("force_log", False),
        obstacle_mask_bool=obstacle_mask_bool, rng=rng)
