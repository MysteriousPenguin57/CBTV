"""Thin runner: orchestrates the CBTV detection->tracking pipeline.

The frame loop here is intentionally just wiring; every real step lives in a
focused module (detection, kernels, cost_matrix, assignment, pkt, bayes, ...).
"""
import csv
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import List

import cv2
import numpy as np

from . import gpu
from .assignment import (
    greedy_confident,
    greedy_confident_sparse,
    hungarian,
    solve_2x2_blocks,
    solve_2x2_blocks_sparse,
    solve_hungarian_stage3,
)
from .color import batch_compute_colors_gpu
from .config import build_parser, inoutflow_active, validate_args
from .cost_matrix import compute_cost_matrix_gpu
from .detect_utils import apply_edge_fit_refinement_2c, get_relaxed_filter_params, radius_based_adaptive_dedup_2c
from .detection import user_detection_with_bayes_adapt
from .detection_filter import _local_stats, _multichannel_stats, cap_candidates, filter_detections_multistage
from .detectors import run_detector
from .fp_memory import FalsePositiveMemory
from .gpu import cp, init_gpu
from .kernels import batch_build_kernels_gpu, prebuild_all_kernels_parallel, precompute_neighbor_lists
from .log_detect import blob_log_gpu_fast
from .pkt import predictive_kernel_early_lock
from .recovery import batch_kernel_recovery
from .tracks import KernelCache, Track, TrackArrays

# T046: the auto-radius rule and its clamps were calibrated on LoG radii (sigma*sqrt2; 4.5 px on PRIMARY).
# CSV radii are true particle radii (PRIMARY: frame-0 median 6.555 px, Cellpose detections of
# data/results/p6_baselines/detections_cellpose/PRIMARY_step1_detections.csv), so they are rescaled into LoG units first.
PRIMARY_CSV_MEDIAN_R = 6.555
CSV_TO_LOG_RADIUS = 4.5 / PRIMARY_CSV_MEDIAN_R


def _auto_radius_radii(frame, args):
    """Frame-0 particle radii for the auto-radius rule, and their source (T046, D30 fix B).

    --detector csv: the `r` of the CSV's processed-frame-0 rows (the detections every tracker shares), no LoG,
    rescaled into LoG-radius units (x CSV_TO_LOG_RADIUS) so the rule's 4.5 px reference and clamps apply unchanged.
    Otherwise: ONE LoG implementation on every device (the CPU skimage path, use_gpu=False), sigma * sqrt(2), so a
    CPU and a GPU run take the same decision (the GPU LoG kernel scales its threshold differently)."""
    if getattr(args, "detector", None) == "csv" and getattr(args, "detections_csv", None):
        from .csv_detector import _load as _csv_load
        rows = _csv_load(args.detections_csv).get(0, [])
        return np.array([d[2] for d in rows], dtype=np.float64) * CSV_TO_LOG_RADIUS, "csv"
    gray_f0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    blobs_f0 = blob_log_gpu_fast(gray_f0, min_sigma=1.5, max_sigma=10.0, num_sigma=10, threshold=0.03,
                                 use_gpu=False)
    if hasattr(blobs_f0, "get"):
        blobs_f0 = blobs_f0.get()
    blobs_f0 = np.asarray(blobs_f0)
    return (blobs_f0[:, 2] * 1.414 if len(blobs_f0) else np.zeros(0)), "log"


def _auto_radius_rule(radii):
    """The frame-0 auto-radius decision (unchanged rule, T046 only moved it): None if <= 10 radii, else a dict with
    r_med/r_p25/r_p75, scale_ratio vs the primary video's 4.5 px, `changed`, and the new r_min/r_max if changed.

    [Phase 4B] Compare detected radius to expected for primary video (~4.5px LoG radius). If significantly different,
    the video has a different pixel scale. [Decision Rationale] Thresholds 0.75/1.5 (Flowchart 12 - Auto-Radius/Init):
    asymmetric because under-estimation (small particles, e.g. K04 at 0.47x) is more dangerous than over-estimation.
    Below 0.75x, default r_min=6 would miss most particles entirely; above 1.5x, particles overflow the kernel and NCC
    degrades. PKT is auto-disabled at non-standard scales (cached templates are calibrated for the primary radius)."""
    radii = np.asarray(radii, dtype=np.float64)
    if len(radii) <= 10:
        return None
    r_med = float(np.median(radii))
    r_p25 = float(np.percentile(radii, 25))
    r_p75 = float(np.percentile(radii, 75))
    expected_r_primary = 4.5
    scale_ratio = r_med / expected_r_primary
    out = {"r_med": r_med, "r_p25": r_p25, "r_p75": r_p75, "scale_ratio": scale_ratio, "changed": False}
    if scale_ratio < 0.75 or scale_ratio > 1.5:
        auto_rmin = max(3, int(r_p25 - 1.0))
        auto_rmax = min(12, int(r_p75 + 2.0) + 1)
        if auto_rmax - auto_rmin < 4:
            auto_rmax = min(12, auto_rmin + 4)
        out.update(changed=True, r_min=auto_rmin, r_max=auto_rmax)
    return out


def main():
    """Entry point for the CBTV hybrid particle tracker.

    Parses the CLI, initialises the GPU, loads the input video and
    optional obstacle mask, estimates particle radii when auto-radius is
    enabled, then runs the per-frame tracking pipeline to completion.
    Writes an annotated output video and a CSV of track states; optionally
    writes per-frame profiling data and diagnostic detection counts.

    Per-frame pipeline summary
    --------------------------
    1. **Detect** — ``user_detection_with_bayes_adapt`` runs Hough
       exemplar generation (and LoG supplement/fallback) → per-channel NCC
       correlation → peak extraction → optional Bayesian refinement.
    2. **Filter** — ``filter_detections_multistage`` removes boundary
       detections, sparse-region FPs, and obstacle-mask violations;
       ``radius_based_adaptive_dedup_2c`` deduplicates overlapping peaks.
    3. **Cap** — ``cap_candidates`` limits the candidate pool when
       enabled (``--cap-mode``).
    4. **Colour extract** — ``compute_particle_colours_lab`` or its GPU
       batch variant extracts per-detection LAB colours.
    5. **PKT early-lock** — ``predictive_kernel_early_lock`` attempts to
       greedily match high-confidence tracks before the global assignment
       step using cached NCC templates.
    6. **Cost matrix** — ``compute_cost_matrix_gpu`` combines spatial
       distance (weight ``--w-dist``) and LAB colour distance (weight
       ``--w-color``) into a joint cost matrix.  Persistence bonus and
       velocity prediction are applied here when enabled.
    7. **3-stage assignment** — ``greedy_confident`` (Stage 1),
       ``greedy_confident_stage2`` (Stage 2), and
       ``solve_hungarian_stage3`` (Stage 3) progressively assign
       remaining unmatched tracks and detections.
    8. **Kernel recovery** — ``kernel_recovery_search`` attempts to
       re-locate missed tracks using correlation-based search in an
       expanded radius.
    9. **Spawn / prune** — Unmatched detections are spawned as new
       tracks; tracks exceeding ``--miss-tolerance`` are pruned.
    10. **Render** — Annotated frame is appended to the output video and
        current track states are written to the output CSV.

    CLI arguments (key subset)
    --------------------------
    --input: Path to the input video file.
    --output-video: Path for the annotated output MP4/AVI.
    --output-csv: Path for the per-frame track CSV.
    --use-gpu: 1 to enable GPU (CuPy + OpenCV CUDA); 0 for CPU fallback.
    --gpu-device: CUDA device index (0 = RTX 3060, 1 = RTX 4070).
    --r-min / --r-max: Particle radius search range in pixels (≥6 / ≤12).
    --threshold: NCC peak acceptance threshold (default 0.45).
    --miss-tolerance: Frames a track can go unmatched before pruning.
    --obstacle-mask: Path to a greyscale obstacle mask image.
    --profile-csv: Optional path to write per-frame timing breakdowns.
    --diagnostic: Log per-stage detection counts to diagnostic_log.csv.

    Pipeline context:
        Called by: ``__main__`` block
        Calls: init_gpu, user_detection_with_bayes_adapt,
               filter_detections_multistage,
               radius_based_adaptive_dedup_2c, cap_candidates,
               compute_particle_colours_lab,
               predictive_kernel_early_lock,
               compute_cost_matrix_gpu, greedy_confident,
               greedy_confident_stage2, solve_hungarian_stage3,
               kernel_recovery_search
        Stage: main_loop

    Returns:
        None.  All output is written to files specified by CLI arguments.

    Phase history:
        - Original: Core detect-assign-spawn-prune loop.
        - Phase 3: LoG fallback detection integrated.
        - Phase 4B: Auto-radius estimation from first-frame Hough results.
        - Phase 4C: PKT early-lock step added (currently disabled due to
          coordinate-system bug — see CLAUDE.md).
        - Phase 5 D: Persistence bonus and last_det_pos tracking.
        - Phase 5 F: Velocity prediction mode (``--use-velocity 1``).
        - Phase 5 G2: Obstacle mask loading, erosion, and pre-detection
          NCC suppression.
        - Phase 5 H: Small-image mode auto-detection and parameter scaling.
        - Phase 6: Per-stage timing instrumentation; PKT lock-rate profiling.
    """
    # === main() section map (see Flowcharts 2, 19, 20) ===
    # The frame loop body walks through twelve named phases labelled
    # [MAIN-1] .. [MAIN-12]. For the pipeline overview see Flowchart 2
    # (current pipeline), Flowchart 19 (original-vs-current diff), and
    # Flowchart 20 (frame-loop function call graph). Each phase references
    # the detailed flowchart for its subsystem.
    #
    # === [MAIN-1] Argparse + config load (see Flowchart 2) ===
    args = build_parser().parse_args()
    # [Phase 3 Stage A] Enforce the explicit cost-weight budget
    # (w_dist + w_color + w_kernel == 1) and flag dependencies.
    validate_args(args)

    # === [Phase 6i Part A] --nd-gates: non-dimensionalise the px limits ===
    # Derive r_min/r_max/max_dist from the calibrated px/diam (--nd-dpx,
    # supplied by the driver from calibration scale x GT diameter — the
    # publish-protocol convention). Ratios MATCH LEGACY at 12.8 px/diam
    # (6/12.8, 10/12.8, 20/12.8), so the flag is a no-op there by
    # construction. The motion floor (max_dist >= --nd-km x m_est, m_est =
    # median coarse-PIV magnitude of the first frame pair) is applied inside
    # the loop once m_est exists; gate values are printed for the audit table.
    if getattr(args, 'nd_gates', 0) == 1:
        _nd_dpx = float(getattr(args, 'nd_dpx', 0.0))
        if _nd_dpx <= 0.0:
            print("[ND-GATES] --nd-dpx required (>0) with --nd-gates 1"); sys.exit(1)
        # Anchor = the MEASURED matched-scale diameter (11.7 px: PRIMARY/M/
        # RES-1080/A_4K all 11.5-12.1 by the publish convention; the nominal
        # 12.8 was 51.2 x 0.25 but the median GT diameter is ~0.228). Within
        # +-6% of the anchor the legacy integer parameters are already the
        # correct rounding -> use them EXACTLY (the pre-registered no-op at
        # matched scale, byte-identical by construction).
        _nd_ref = 11.7
        if abs(_nd_dpx / _nd_ref - 1.0) <= 0.06:
            print(f"[ND-GATES] dpx={_nd_dpx:.2f} within 6% of anchor {_nd_ref} "
                  f"-> LEGACY params kept (no-op)", flush=True)
        else:
            args.r_min = max(2, int(round(6.0 / _nd_ref * _nd_dpx)))
            args.r_max = max(args.r_min + 2, int(round(10.0 / _nd_ref * _nd_dpx)))
            args.max_dist = round(20.0 / _nd_ref * _nd_dpx, 2)
            print(f"[ND-GATES] dpx={_nd_dpx:.2f} -> r_min={args.r_min} "
                  f"r_max={args.r_max} max_dist={args.max_dist} "
                  f"(motion floor km={getattr(args, 'nd_km', 1.25)} pending m_est)",
                  flush=True)


    # === [MAIN-2] GPU init + video open + obstacle mask load (see Flowcharts 2, 12) ===
    # GPU context, video reader, optional obstacle mask (with erosion buffer
    # from Phase 5-G2) are set up once before the frame loop begins.
    init_gpu(args.use_gpu, args.gpu_device)
    # Every CuPy path below keys off args.use_gpu: follow the device init_gpu actually got, so a GPU request on a
    # machine without CUDA (CuPy missing or no device) runs the CPU path instead of crashing.
    # OpenCL is left as init_gpu set it.
    if args.use_gpu and not gpu.USE_CUPY:
        args.use_gpu = 0
    # One generator for every detection-path RNG draw: runs are
    # reproducible for a given --seed instead of depending on global state.
    det_rng = np.random.default_rng(args.seed)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): print("Cannot open video:", args.input); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Input-polarity normalisation. Off by default: with --input-polarity
    # dark this is None, nothing is imported and every frame reaches the
    # detector untouched.
    polariser = None
    if getattr(args, "input_polarity", "dark") != "dark":
        from .polarity import make_normaliser
        polariser = make_normaliser(cap, args.input_polarity, r_max=args.r_max)

    # Workstream G2: load obstacle mask
    obstacle_mask_bool = None
    if args.obstacle_mask and os.path.isfile(args.obstacle_mask):
        _omask = cv2.imread(args.obstacle_mask, cv2.IMREAD_GRAYSCALE)
        if _omask is not None:
            if _omask.shape[:2] != (H, W):
                _omask = cv2.resize(_omask, (W, H), interpolation=cv2.INTER_NEAREST)
            obstacle_mask_bool = _omask > 127
            # Workstream G2+: erode mask to create buffer zone (eliminates boundary FPs)
            if args.mask_erode > 0:
                erode_kernel = np.ones((2 * args.mask_erode + 1, 2 * args.mask_erode + 1), dtype=np.uint8)
                obstacle_mask_bool = cv2.erode(obstacle_mask_bool.astype(np.uint8), erode_kernel, iterations=1).astype(bool)
                print(f"[MASK] Eroded by {args.mask_erode}px (kernel {2*args.mask_erode+1}x{2*args.mask_erode+1})")
            print(f"[MASK] Obstacle mask loaded: {args.obstacle_mask}")
            print(f"[MASK] Valid region: {np.count_nonzero(obstacle_mask_bool)}/{obstacle_mask_bool.size} "
                  f"({np.count_nonzero(obstacle_mask_bool)/obstacle_mask_bool.size*100:.1f}%)")
        else:
            print(f"[MASK] WARNING: Could not load mask from {args.obstacle_mask}")
    elif args.obstacle_mask:
        print(f"[MASK] WARNING: Mask file not found: {args.obstacle_mask}")

    # === [MAIN-3] Small-image mode + output writer + CSV setup (see Flowchart 12) ===
    # Phase 5-H auto-triggers for slit viewports (L03); scaled detection
    # parameters are wired into args before the loop starts. The VideoWriter
    # and CSV header are opened right after so the frame loop can stream directly.
    # [Phase 5-H] Small-image mode detection: for videos like L03 (624x128 slit viewport)
    small_image_mode = False
    if getattr(args, 'small_image_mode', 'auto') != 'off':
        if getattr(args, 'small_image_mode', 'auto') == 'on' or min(H, W) < 256:
            small_image_mode = True
            print(f"[SMALL-IMAGE] Activated: {W}x{H} (min dim {min(H,W)})")
            sim_scale_factor = min(H, W) / 1024.0
            # Force LoG detection (Hough can't handle small images)
            args.force_log = True
            # Lower sparse filter threshold — scene genuinely has few particles
            args.sparse_filter_threshold = max(3, int(23 * sim_scale_factor))
            # Shrink edge exclusion zone
            args.edge_distance = max(2, int(getattr(args, 'edge_distance', 5) * sim_scale_factor))
            # Relax max_dist for smaller coordinate space
            args.max_dist = max(5.0, getattr(args, 'max_dist', 15.0) * sim_scale_factor * 2)
            # Force smaller radius for small-image particles
            # L03 particles are ~2-4px radius; scale from primary's 6-10 range
            # B07: never for --detector csv - the radii come with the detections, and the forced 2-4 range
            # demoted / deduplicated r~7 csv detections; otherwise the override is announced, not silent.
            if args.detector == "csv":
                print("[SMALL-IMAGE] --detector csv: keeping r_min/r_max "
                      f"({args.r_min}/{args.r_max}); radii come from the detections")
            elif args.r_min == 6 or args.r_max == 10:
                r_min0, r_max0 = args.r_min, args.r_max
                args.r_min = max(2, int(args.r_min * sim_scale_factor + 0.5)) if args.r_min == 6 else args.r_min
                args.r_max = max(4, int(args.r_max * sim_scale_factor * 1.5 + 0.5)) if args.r_max == 10 else args.r_max
                print(f"[WARN] small-image mode: radius range {r_min0}-{r_max0} px (default) rescaled to "
                      f"{args.r_min}-{args.r_max} px for this {W}x{H} video; pass other --r-min/--r-max to keep "
                      "yours, or --small-image-mode off")
            if args.r_max - args.r_min < 2:
                args.r_max = args.r_min + 2
            print(f"[SMALL-IMAGE] scale_factor={sim_scale_factor:.3f}, max_dist={args.max_dist:.1f}, "
                  f"sparse_threshold={args.sparse_filter_threshold}, edge_dist={args.edge_distance}, "
                  f"r_min={args.r_min}, r_max={args.r_max}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (W,H))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        alt = args.output_video if args.output_video.lower().endswith(".avi") else args.output_video.replace(".mp4",".avi")
        writer = cv2.VideoWriter(alt, fourcc, fps, (W,H))

    csv_fp = open(args.output_csv, "w", newline=""); csv_wr = csv.writer(csv_fp)

    # [In/outflow] region map + tagged/delayed row writer (cbtv.inoutflow).
    # With every in/outflow flag at its default, _io_region and _io_rows stay
    # None, the module is never imported and the legacy writer below runs.
    _io_region = None
    _io_rows = None
    _io_tags = getattr(args, 'row_tags', 0) == 1
    _io_on = inoutflow_active(args)          # any lifecycle flag (A-E) set
    _io_K = int(getattr(args, 'confirm_hits', 1))
    _io_Ke = int(getattr(args, 'confirm_hits_entry', 0)) or _io_K
    _io_exit = getattr(args, 'exit_kill', 0) == 1
    _io_occN = int(getattr(args, 'occluder_coast', 0))
    _io_coast = getattr(args, 'coast_predict', 'freeze')
    if _io_tags or _io_on:
        from .inoutflow import EXIT as _IO_EXIT
        from .inoutflow import OCCLUDER as _IO_OCC
        from .inoutflow import RegionMap, RowWriter, meta_path_for
        from .inoutflow import birth_reason as _io_birth_reason
        _io_region = RegionMap(W, H, getattr(args, 'region_map', ''),
                               getattr(args, 'region_transform', ''),
                               getattr(args, 'region_transform_mode', 'camera'))
        _io_rows = RowWriter(csv_wr, row_tags=_io_tags, delay=max(_io_K, _io_Ke) - 1,
                             region=_io_region,
                             r_from_det=(getattr(args, 'csv_trust', 0) == 1
                                         and getattr(args, 'detector', '') == 'csv'),
                             meta_path=meta_path_for(args.output_csv) if _io_tags else "")
        if _io_on:
            print(f"[IO] coast={_io_coast} exit_kill={int(_io_exit)} "
                  f"(margin {getattr(args, 'exit_margin', 1.0)} r) occluder_coast={_io_occN} "
                  f"(gate {getattr(args, 'occluder_gate', 2.0)} r) confirm_hits={_io_K}/"
                  f"entry {_io_Ke} row_tags={int(_io_tags)}")
    else:
        csv_wr.writerow(["frame","track_id","x","y","r","missed"])

    # [Phase 6g Stage 1] link-diagnostics dump (write-only observer; flag off
    # -> _linkdump_wr is None and no per-frame diagnostic code executes).
    _linkdump_fp = None
    _linkdump_wr = None
    if getattr(args, 'debug_linkdump', ''):
        _linkdump_fp = open(args.debug_linkdump, "w", newline="")
        _linkdump_wr = csv.writer(_linkdump_fp)
        _linkdump_wr.writerow(["frame", "track_id", "prev_x", "prev_y",
                               "pred_x", "pred_y", "src", "alpha", "has_vel",
                               "missed_before", "matched", "stage",
                               "det_x", "det_y", "gate_dist",
                               "n_cand_gate", "n_cand_samecol"])

    # === [MAIN-4] Track state + profiling accumulators init (see Flowcharts 10, 20) ===
    # First-frame seed tracks are appended inside the frame loop (spawn path);
    # here we just initialise the empty track list, kernel cache, FalsePositive
    # memory, multi-frame ring buffer, and the Phase 6 profiling counters.
    tracks: List[Track] = []; next_id=1
    kernel_cache = KernelCache()  # Simplified cache - kernels rebuilt each frame

    # Adaptive search now uses simple inline KD-tree filtering (no complex manager needed)
    if args.use_adaptive_kernel_search:
        print("[OPTIMIZATION] Simple KD-Tree spatial filtering enabled")

    prof_wr=None
    if args.profile_csv:
        pf=open(args.profile_csv,"w",newline=""); prof_wr=csv.writer(pf)
        # [Phase 0 V0] Added: t_pkt, n_pairs_total, n_pairs_gated_dist,
        # n_pairs_gated_combined, mean_cands_per_track, n_kernel_evals,
        # pkt_full_locks, pkt_soft_locks, pkt_low_ncc_skipped,
        # pkt_backwards_locks. These let Phase-3 decide whether tracking
        # kernels, detection, or Bayes dominate the per-frame budget.
        # [Phase 2] Added field columns: field_engaged_tracks (tracks for which
        # alpha > 0), field_alpha_mean (mean over engaged tracks),
        # field_easy_committed (Tier-0 easy-match commits), field_anchors_mean
        # (mean anchors used per fitted track).
        prof_wr.writerow(["frame","raw","bayes_added","deduped","kept","soft","tracks","matched_s1","matched_s2","matched_s3","spawned","sp2","krec","thr",
                         "t_det","t_filt","t_cap","t_color_extract","t_pkt",
                         "t_dist_matrix","t_color_matrix","t_kernel_build","t_kernel_corr",
                         "kernel_hits","kernel_misses",
                         "n_pairs_total","n_pairs_gated_dist","n_pairs_gated_combined",
                         "mean_cands_per_track","n_kernel_evals",
                         "pkt_full_locks","pkt_soft_locks","pkt_low_ncc_skipped",
                         "pkt_backwards_locks",
                         "field_engaged_tracks","field_alpha_mean",
                         "field_easy_committed","field_anchors_mean",
                         # [Phase 3 Stage C] constellation diagnostics
                         "const_proposed","const_accepted","const_members",
                         "const_outliers","const_field_refreshed","t_const",
                         "t_asn","t_sp2","t_krec","total"])

    # [Phase 3 Stage C] Per-commit constellation log (member-accuracy audit).
    # [Phase 4 Stage 2b] Track ids constellation-committed in the previous
    # frame (temporal field propagation; empty set = no effect).
    _const_committed_prev: set = set()
    _cluster_log_fp = None
    _cluster_log_wr = None
    if getattr(args, 'use_constellations', 0) == 1:
        _cl_path = os.path.join(os.path.dirname(args.output_csv) or ".", "cluster_log.csv")
        _cluster_log_fp = open(_cl_path, "w", newline="")
        _cluster_log_wr = csv.writer(_cluster_log_fp)
        _cluster_log_wr.writerow(["frame", "cluster_id", "track_id",
                                  "det_x", "det_y", "residual_px", "conf", "committed"])

    # --- Diagnostic mode setup ---
    diag_wr = None
    diag_fp = None
    diag_gt_counts = {}
    if args.diagnostic:
        diag_fp = open("diagnostic_log.csv", "w", newline="")
        diag_wr = csv.writer(diag_fp)
        diag_wr.writerow(["frame", "hough_raw", "after_nms", "after_lda", "after_edgefit",
                          "after_filter", "gt_count", "lda_score_mean", "lda_score_std", "lda_score_min"])
        # Load ground truth counts per frame (type-3 atoms from blob_data 3.txt)
        gt_path = os.path.join(os.path.dirname(args.input), "blob_data 3.txt")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r') as gf:
                    gt_lines = [ln.strip() for ln in gf]
                cur_ts = -1
                in_atoms = False
                for i, line in enumerate(gt_lines):
                    if line.startswith("ITEM: TIMESTEP"):
                        cur_ts = int(gt_lines[i + 1])
                        diag_gt_counts[cur_ts] = 0
                        in_atoms = False
                    elif line.startswith("ITEM: ATOMS"):
                        in_atoms = True
                    elif in_atoms and line and not line.startswith("ITEM:"):
                        parts = line.split()
                        if len(parts) >= 4 and int(parts[1]) == 3:
                            diag_gt_counts[cur_ts] = diag_gt_counts.get(cur_ts, 0) + 1
                    elif in_atoms and line.startswith("ITEM:"):
                        in_atoms = False
                print(f"[DIAGNOSTIC] Loaded GT counts for {len(diag_gt_counts)} frames from {gt_path}")
            except Exception as e:
                print(f"[DIAGNOSTIC] Could not load GT: {e}. Using -1 for gt_count.")
        else:
            print(f"[DIAGNOSTIC] GT file not found at {gt_path}. Using -1 for gt_count.")

    thr=args.threshold
    processed=0
    _pkt_auto_disable = False  # Set True when auto-radius detects non-standard scale

    previous_frame = None
    prev_lab = None # Initialize prev_lab here
    previous_detections = None

    # Catastrophic drop recovery tracking
    detection_history = deque(maxlen=args.recovery_history_size)
    prev_detection_params = None  # Store (gray, chans, dbg) for kernel recovery

    # Frame buffer for multi-frame detection (simple rolling window)
    frame_buffer_gray = []
    max_buffer_size = 2 * args.multiframe_offset + 1 if args.multiframe_detection else 0

    # Initialize False Positive Memory for improved Bayesian refinement
    fp_memory = FalsePositiveMemory(max_history_frames=10, max_negatives_per_frame=50)

    if args.verbose and args.multiframe_detection:
        print(f"[Multi-frame] Using rolling buffer (size: {max_buffer_size}) for detection kernel improvement")

    # Phase 6 profiling accumulators
    _prof_pkt_full_lock_total = 0
    _prof_pkt_soft_lock_total = 0
    _prof_pkt_skip_total = 0
    _prof_pkt_active_total = 0
    _prof_hungarian_rows_total = 0
    _prof_hungarian_cols_total = 0
    _prof_hungarian_frames = 0
    _prof_no_pkt_rows_total = 0  # What Hungarian size would be without PKT
    _prof_no_pkt_cols_total = 0
    # [Phase 0] PKT backwards-lock counter — magnitude of the documented "locks tracks
    # backwards" bug. Incremented inside predictive_kernel_early_lock when a successful
    # lock moves the track in a direction opposite to its prior velocity.
    _prof_pkt_backwards_lock_total = 0

    if args.progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0, desc="Hybrid Tracking (GPU)" if args.use_gpu else "Hybrid Tracking")
        except ImportError:
            print("[WARN] tqdm not installed: no progress bar (results unaffected)")
            pbar=None

    else:
        pbar=None

    # === [MAIN-5] Frame loop entry (see Flowcharts 2, 20) ===
    # Each iteration processes exactly one video frame through phases
    # MAIN-6 (detect) -> MAIN-7 (PKT) -> MAIN-8 (cost) -> MAIN-9 (assign)
    # -> MAIN-10 (lifecycle) -> MAIN-11 (render). MAIN-12 runs once after.
    # Phase 3f: --frame-step support. The argparse flag was already
    # plumbed (line 5688) but main()'s read loop ignored it (the
    # FrameReader thread that respects it is never instantiated). Honour
    # it here by discarding (frame_step - 1) source frames before each
    # processed frame. Default frame_step=1 -> no skip, identical to
    # prior behaviour.
    _frame_step = max(1, int(getattr(args, 'frame_step', 1)))
    _first_iter = True
    while True:
        t0=time.perf_counter()
        # [Phase 3l fix 2026-04-30] Read first, skip after. The previous
        # version skipped (frame_step-1) frames BEFORE the first read,
        # which made the first kept frame = source frame (frame_step-1)
        # instead of source frame 0. At step=4 the tracker actually
        # processed source frames 3,7,11,... while the evaluator's
        # frame_to_ts mapping (and our remap_csv_for_step multiplier)
        # expected source frames 0,4,8,... -> ~3-frame source offset
        # produced the systematic ~3px median position bias seen across
        # all trackers using the linker. Fix: read first, then skip
        # AFTER the read for subsequent iterations.
        if not _first_iter:
            for _skip in range(_frame_step - 1):
                ok_skip, _ = cap.read()
                if not ok_skip:
                    break
        _first_iter = False
        ret, frame = cap.read()
        if not ret: break

        # Normalise input polarity before anything reads the frame, so
        # detection, the LAB colour path and the overlay all see the same
        # image. No-op unless --input-polarity selected it.
        if polariser is not None:
            frame = polariser.apply(frame)

        # [In/outflow A] region transform follows the SOURCE frame index
        # (read-first-then-skip: processed frame k is source frame k*step).
        if _io_region is not None:
            _io_region.set_frame(processed * _frame_step)

        # Simple rolling buffer - just add current frame
        if args.multiframe_detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
            frame_buffer_gray.append(gray)
            # Keep only last max_buffer_size frames
            if len(frame_buffer_gray) > max_buffer_size:
                frame_buffer_gray.pop(0)

        # Phase 1: Single LAB Conversion
        # prev_frame_lab is used for kernel correlation, so it needs to be the LAB of the *previous* BGR frame
        # If prev_lab is None (first frame), we need to compute it from prev_frame (which is also None, so this needs careful handling)
        # For the first frame, prev_frame_lab should effectively be skipped or handled as an empty/dummy array.
        # The compute_cost_matrix_gpu function already handles prev_frame being None.
        if prev_lab is not None:
            prev_frame_lab_gpu = cp.asarray(prev_lab) if args.use_gpu else prev_lab
            # L_img = prev_frame_lab_gpu[..., 0] # These are not used directly here, only passed to compute_cost_matrix_gpu
            # A_img = prev_frame_lab_gpu[..., 1]
            # B_img = prev_frame_lab_gpu[..., 2]
        else:
            prev_frame_lab_gpu = None # No previous LAB frame available yet

        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        if args.use_gpu:
            frame_lab_gpu = cp.asarray(frame_lab)
        else:
            frame_lab_gpu = None

        dump = None


        if processed==0 and args.dump_first_frame_corr:
            dump = dict(dir=args.dump_dir, save_corr=1)

        # ==================================================================
        # AUTO-RADIUS ESTIMATION (frame 0 only, if user didn't override)
        # ==================================================================
        if processed == 0 and args.r_min == 6 and args.r_max == 10:
            # T046: radii from the frame-0 CSV rows with --detector csv; else ONE LoG (CPU) on every device.
            radii_f0, _ar_src = _auto_radius_radii(frame, args)
            _ar = _auto_radius_rule(radii_f0)
            if _ar is not None:
                r_med, r_p25, r_p75, scale_ratio = _ar["r_med"], _ar["r_p25"], _ar["r_p75"], _ar["scale_ratio"]
                if _ar["changed"]:
                    auto_rmin, auto_rmax = _ar["r_min"], _ar["r_max"]
                    print(f"[AUTO-RADIUS] source={_ar_src} Scale change detected ({scale_ratio:.2f}x): "
                          f"median_r={r_med:.1f} [p25={r_p25:.1f}, p75={r_p75:.1f}], "
                          f"setting r_min={auto_rmin}, r_max={auto_rmax} "
                          f"(was r_min={args.r_min}, r_max={args.r_max})")
                    args.r_min = auto_rmin
                    args.r_max = auto_rmax
                    # PKT templates are unreliable at non-standard scales
                    _pkt_auto_disable = True
                    print(f"[AUTO-RADIUS] PKT auto-disabled (scale_ratio={scale_ratio:.2f})")
                else:
                    print(f"[AUTO-RADIUS] source={_ar_src} Defaults confirmed: r_min={args.r_min}, r_max={args.r_max} "
                          f"(median_r={r_med:.1f}, scale_ratio={scale_ratio:.2f})")
            else:
                print(f"[AUTO-RADIUS] source={_ar_src} Too few blobs ({len(radii_f0)}) for estimation, "
                      f"keeping defaults")

        # ==================================================================
        # === [MAIN-6] Detection pipeline call (see Flowcharts 6, 11, 14, 17) ===
        # Runs user_detection_with_bayes_adapt (Hough exemplars + LoG supplement,
        # Bayesian refinement via FalsePositiveMemory), followed by multi-stage
        # filtering, adaptive dedup, edge-fit refinement, and the final filter
        # pass that feeds the FP memory for the next frame. All downstream
        # phases operate on `kept` and `soft` produced here.
        # DETECTION PHASE (see Flowchart 6: LoG Integration, Flowchart 11: Filter Pipeline)
        # ==================================================================
        t_det0=time.perf_counter()

        # Stage 1: Detection with multi-frame candidate pooling
        dets0, dets_bayes_added, gray, chans, dbg = run_detector(
            args.detector, frame, args, thr=thr, dump=dump,
            frame_buffer_gray=frame_buffer_gray, current_frame_idx=processed,
            fp_memory=fp_memory, obstacle_mask_bool=obstacle_mask_bool, rng=det_rng)
        t_det_stage1 = time.perf_counter()

        # Catastrophic drop recovery
        _csv_trust = (getattr(args, 'csv_trust', 0) == 1
                      and getattr(args, 'detector', '') == 'csv')
        if args.catastrophic_recovery and not _csv_trust:
            current_count = len(dets0) + len(dets_bayes_added)
            detection_history.append(current_count)

            # Only check after we have enough history (minimum 3 frames)
            if len(detection_history) >= 3:
                moving_avg = sum(detection_history) / len(detection_history)

                # Detect catastrophic drop (current count < threshold * moving average)
                if current_count < args.recovery_drop_threshold * moving_avg:
                    if args.verbose:
                        print(f"Frame {processed}: Drop detected ({current_count} < {args.recovery_drop_threshold*moving_avg:.1f})")

                    # Cascading recovery attempts
                    recovery_succeeded = False

                    # Step 1: Try with previous frame's parameters if available
                    if prev_detection_params is not None and not recovery_succeeded:
                        try:
                            prev_gray, prev_chans, prev_dbg = prev_detection_params
                            # Use previous frame's image features with current threshold
                            dets_rec0, dets_rec_bayes, _, _, _ = user_detection_with_bayes_adapt(
                                frame, args.r_min, args.r_max, thr, topk=args.topk_channels,
                                nms_factor=args.nms_factor, hough_polarity=args.hough_polarity,
                                refine_bayes=args.refine_bayes, refine_top_pos=args.refine_top_pos,
                                refine_top_neg=args.refine_top_neg, refine_beta=args.refine_beta,
                                thr_mode=args.thr_mode, thr_local_k=args.thr_local_k, thr_local_ksize=args.thr_local_ksize,
                                thr_topk_tile=args.thr_topk_tile, thr_tile_factor=args.thr_tile_factor,
                                dedup_threshold=args.dedup_threshold, intelligent_nms=args.intelligent_nms, dump=None,
                                frame_buffer_gray=frame_buffer_gray if args.multiframe_detection else None,
                                current_frame_idx=processed,
                                multiframe_offset=args.multiframe_offset,
                                fp_memory=fp_memory,
                                enable_subpixel=bool(args.subpixel_refinement),
                                use_antialias=bool(args.antialias_kernels),
                                use_simpson=bool(args.better_integration),
                                opt_gpu_blob=args.opt_gpu_blob,
                                force_log=getattr(args, 'force_log', False),
                                obstacle_mask_bool=obstacle_mask_bool, rng=det_rng)

                            rec_count = len(dets_rec0) + len(dets_rec_bayes)
                            if rec_count >= args.recovery_drop_threshold * moving_avg:
                                dets0, dets_bayes_added = dets_rec0, dets_rec_bayes
                                recovery_succeeded = True
                                if args.verbose:
                                    print(f"  Recovery Step 1: Previous params recovered {rec_count} detections")
                        except Exception as e:
                            if args.verbose:
                                print(f"  Recovery Step 1 failed: {e}")

                    # Step 2: Lower threshold slightly (threshold * 0.87)
                    if not recovery_succeeded:
                        try:
                            lower_thr = max(args.threshold_min if args.adaptive_threshold else 0.3, thr * 0.87)
                            dets_rec0, dets_rec_bayes, _, _, _ = user_detection_with_bayes_adapt(
                                frame, args.r_min, args.r_max, lower_thr, topk=args.topk_channels,
                                nms_factor=args.nms_factor, hough_polarity=args.hough_polarity,
                                refine_bayes=args.refine_bayes, refine_top_pos=args.refine_top_pos,
                                refine_top_neg=args.refine_top_neg, refine_beta=args.refine_beta,
                                thr_mode=args.thr_mode, thr_local_k=args.thr_local_k, thr_local_ksize=args.thr_local_ksize,
                                thr_topk_tile=args.thr_topk_tile, thr_tile_factor=args.thr_tile_factor,
                                dedup_threshold=args.dedup_threshold, intelligent_nms=args.intelligent_nms, dump=None,
                                frame_buffer_gray=frame_buffer_gray if args.multiframe_detection else None,
                                current_frame_idx=processed,
                                multiframe_offset=args.multiframe_offset,
                                fp_memory=fp_memory,
                                enable_subpixel=bool(args.subpixel_refinement),
                                use_antialias=bool(args.antialias_kernels),
                                use_simpson=bool(args.better_integration),
                                opt_gpu_blob=args.opt_gpu_blob,
                                force_log=getattr(args, 'force_log', False),
                                obstacle_mask_bool=obstacle_mask_bool, rng=det_rng)

                            rec_count = len(dets_rec0) + len(dets_rec_bayes)
                            if rec_count >= args.recovery_drop_threshold * moving_avg:
                                dets0, dets_bayes_added = dets_rec0, dets_rec_bayes
                                recovery_succeeded = True
                                if args.verbose:
                                    print(f"  Recovery Step 2: Lower threshold ({lower_thr:.3f}) recovered {rec_count} detections")
                        except Exception as e:
                            if args.verbose:
                                print(f"  Recovery Step 2 failed: {e}")

                    # Step 3: More aggressive threshold (minimum threshold)
                    if not recovery_succeeded:
                        try:
                            aggressive_thr = args.recovery_min_threshold
                            dets_rec0, dets_rec_bayes, _, _, _ = user_detection_with_bayes_adapt(
                                frame, args.r_min, args.r_max, aggressive_thr, topk=args.topk_channels,
                                nms_factor=args.nms_factor, hough_polarity=args.hough_polarity,
                                refine_bayes=args.refine_bayes, refine_top_pos=args.refine_top_pos,
                                refine_top_neg=args.refine_top_neg, refine_beta=args.refine_beta,
                                thr_mode=args.thr_mode, thr_local_k=args.thr_local_k, thr_local_ksize=args.thr_local_ksize,
                                thr_topk_tile=args.thr_topk_tile, thr_tile_factor=args.thr_tile_factor,
                                dedup_threshold=args.dedup_threshold, intelligent_nms=args.intelligent_nms, dump=None,
                                frame_buffer_gray=frame_buffer_gray if args.multiframe_detection else None,
                                current_frame_idx=processed,
                                multiframe_offset=args.multiframe_offset,
                                fp_memory=fp_memory,
                                enable_subpixel=bool(args.subpixel_refinement),
                                use_antialias=bool(args.antialias_kernels),
                                use_simpson=bool(args.better_integration),
                                opt_gpu_blob=args.opt_gpu_blob,
                                force_log=getattr(args, 'force_log', False),
                                obstacle_mask_bool=obstacle_mask_bool, rng=det_rng)

                            rec_count = len(dets_rec0) + len(dets_rec_bayes)
                            # Use best effort even if below threshold
                            dets0, dets_bayes_added = dets_rec0, dets_rec_bayes
                            if args.verbose:
                                print(f"  Recovery Step 3: Aggressive threshold ({aggressive_thr:.3f}) recovered {rec_count} detections")
                        except Exception as e:
                            if args.verbose:
                                print(f"  Recovery Step 3 failed: {e}")

            # Store current detection parameters for next frame's potential recovery
            prev_detection_params = (gray.copy() if gray is not None else None,
                                     chans.copy() if isinstance(chans, dict) else None,
                                     dbg)

        t_det_cat = time.perf_counter()

        # [csv-trust] --csv-trust 1 with --detector csv: the external
        # detections are already validated, so they skip every image-quality
        # stage below (multistage filter incl. SPARSE tightening, obstacle
        # mask, dedup merge, edge-fit refinement, final filter): all are
        # hard (can spawn), positions and radii exactly as given. Default 0
        # runs the stages unchanged.
        if _csv_trust:
            kept0 = list(dets0) + list(dets_bayes_added)
            soft0, killed0 = [], []
            merged = deduped = refined = kept0
            t_det_filter1 = t_det_dedup = t_det_edge = t_det_filter2 = time.perf_counter()
        else:
            # Stage 2: Two-tier filtering strategy
            # RESTORED: Using ORIGINAL parameters (hard=0.6, soft=0.8 per CORRECTED_FINAL_SUMMARY.md)
            #            Pass hard but fail soft → soft_keep (can match tracks only)
            #            Pass both → kept (can spawn new tracks)
            avg_r_all = (args.r_min + args.r_max) / 2.0
            ksz = int(max(3, 2*avg_r_all+1)); ksz += (ksz % 2 == 0)

            if chans is not None:
                var, gmax, gx_channels, gy_channels = _multichannel_stats(chans, ksz)
                gx_list = [gx_channels[name] for name in gx_channels.keys()]
                gy_list = [gy_channels[name] for name in gy_channels.keys()]
                gx_stack = np.stack(gx_list, axis=0)
                gy_stack = np.stack(gy_list, axis=0)
                grad_mags = np.sqrt(gx_stack**2 + gy_stack**2)
                max_idx = np.argmax(grad_mags, axis=0)
                gx = np.take_along_axis(gx_stack, max_idx[None, :, :], axis=0)[0]
                gy = np.take_along_axis(gy_stack, max_idx[None, :, :], axis=0)[0]
                shared_stats = (var, gmax, gx, gy)
            else:
                shared_stats = _local_stats(gray, ksz)

            relaxed_hard, _ = get_relaxed_filter_params(relaxation_factor=args.hard_factor)
            _, relaxed_soft = get_relaxed_filter_params(relaxation_factor=args.soft_factor)
            kept_hough, soft_hough, _ = filter_detections_multistage(gray, dets0,
                                                             hard=relaxed_hard,
                                                             soft=relaxed_soft,
                                                             channels_dict=chans,
                                                             args=args,
                                                             stats=shared_stats,
                                                             small_image_mode=small_image_mode)
            kept_bayes, soft_bayes, _ = filter_detections_multistage(gray, dets_bayes_added,
                                                             hard=relaxed_hard,
                                                             soft=relaxed_soft,
                                                             channels_dict=chans,
                                                             args=args,
                                                             stats=shared_stats,
                                                             small_image_mode=small_image_mode)
            merged = list(kept_hough) + list(kept_bayes) + list(soft_hough) + list(soft_bayes)
            t_det_filter1 = time.perf_counter()

            # Workstream G2: safety-net obstacle mask filter on detection candidates
            if obstacle_mask_bool is not None:
                pre_mask_count = len(merged)
                merged_filtered = []
                for det in merged:
                    x, y = int(round(det[0])), int(round(det[1]))
                    if 0 <= y < obstacle_mask_bool.shape[0] and 0 <= x < obstacle_mask_bool.shape[1]:
                        if obstacle_mask_bool[y, x]:
                            merged_filtered.append(det)
                    else:
                        merged_filtered.append(det)
                if pre_mask_count != len(merged_filtered):
                    print(f"[MASK] Filtered {pre_mask_count - len(merged_filtered)} obstacle detections "
                          f"({len(merged_filtered)} remaining)")
                merged = merged_filtered

            # Stage 3: Radius-based adaptive deduplication (threshold = radius × 1.0)
            # However, if we used LoG, it means we are in a dense region and many particles
            # overlap heavily. Suppressing them at 1.0 radius kills valid ground truth particles.
            radius_mult_use = args.nms_factor if dbg.get('is_log_fallback', False) else 1.0

            deduped = radius_based_adaptive_dedup_2c(merged, gray, chans,
                                                      relaxed_hard, relaxed_soft,
                                                      radius_mult=radius_mult_use,
                                                      stats=shared_stats)
            t_det_dedup = time.perf_counter()

            # Stage 4: Edge-Fit refinement
            refined = apply_edge_fit_refinement_2c(deduped, gray)
            t_det_edge = time.perf_counter()

            # Stage 5: Final filter with RELAXED thresholds (0.6x)
            kept0, soft0, killed0 = filter_detections_multistage(gray, refined,
                                                                  hard=relaxed_hard,
                                                                  soft=relaxed_soft,
                                                                  channels_dict=chans,
                                                                  args=args,
                                                                  stats=shared_stats,
                                                                  small_image_mode=small_image_mode)
            t_det_filter2 = time.perf_counter()

            # Store killed detections for improved Bayesian refinement in future frames
            if processed > 0:  # Don't learn from first frame
                fp_memory.add_frame_negatives(processed, killed0)

            # --- Diagnostic logging ---
            if diag_wr is not None:
                d_hough_raw = len(dets0) + len(dets_bayes_added)
                d_after_lda = len(merged)          # after pre-filter (quality classification)
                d_after_nms = len(deduped)          # after radius_based_adaptive_dedup_2c
                d_after_edgefit = len(refined)      # after edge-fit refinement
                d_after_filter = len(kept0) + len(soft0)  # after final hard/soft filter
                # GT timesteps are frame_idx * 1000 (LAMMPS convention)
                d_gt = diag_gt_counts.get(processed * 1000, -1)
                # LDA score distribution: scores of detections BEFORE final filter
                # Use refined detections (input to final filter) — scores are the 4th element
                if refined:
                    scores_arr = np.array([s for _, _, _, s in refined])
                    d_lda_mean = float(np.mean(scores_arr))
                    d_lda_std = float(np.std(scores_arr))
                    d_lda_min = float(np.min(scores_arr))
                else:
                    d_lda_mean, d_lda_std, d_lda_min = 0.0, 0.0, 0.0
                diag_wr.writerow([processed, d_hough_raw, d_after_nms, d_after_lda,
                                  d_after_edgefit, d_after_filter, d_gt,
                                  f"{d_lda_mean:.6f}", f"{d_lda_std:.6f}", f"{d_lda_min:.6f}"])
                diag_fp.flush()

        t_det=(time.perf_counter()-t_det0)*1000.0
        if processed < 5:
            print(f"[POST_DET_PROF] bayes={(t_det_stage1-t_det0)*1000:.1f}ms filt1={(t_det_filter1-t_det_cat)*1000:.1f}ms dedup={(t_det_dedup-t_det_filter1)*1000:.1f}ms edge={(t_det_edge-t_det_dedup)*1000:.1f}ms filt2={(t_det_filter2-t_det_edge)*1000:.1f}ms")

        t_filt0=time.perf_counter()
        t_filt=(time.perf_counter()-t_filt0)*1000.0

        # CAP
        t_cap0=time.perf_counter()
        kept_sorted = sorted(kept0, key=lambda d: d[3], reverse=True)
        kept, soft = cap_candidates(kept_sorted, soft0, len(tracks), mode=args.cap_mode, cap_min=args.cap_min,
                                    cap_warmup=args.cap_warmup, extra_frac=args.cap_extra_frac, fixed=args.cap_fixed,
                                    ratio=args.cap_ratio, frame_idx=processed)
        t_cap=(time.perf_counter()-t_cap0)*1000.0

        # Adaptive threshold
        if args.adaptive_threshold:
            target = len(tracks) + int(math.ceil(0.15*max(1,len(tracks))))
            err = len(kept)+len(soft) - target
            if err > 0: thr = min(args.threshold_max, thr + args.thr_step)
            elif err < 0: thr = max(args.threshold_min, thr - args.thr_step)

        # ========================================================================
        # === [MAIN-7] PKT early lock (see Flowchart 7) ===
        # Before running the full cost matrix, predictive_kernel_early_lock
        # tries a one-frame NCC template match for each confident track. On
        # primary video this locks ~99.7% of tracks and shrinks the Hungarian
        # problem from ~540x540 to ~5x5. Locked tracks are snapped to the
        # nearest matching detection and removed from the assignment pool.
        # PKT + ASSIGNMENT PHASE (see Flowchart 7: PKT Early Lock, Flowchart 8: Cost Matrix, Flowchart 9: Assignment Pipeline)
        # ========================================================================
        t_asn0=time.perf_counter()

        matched_tracks=set()
        matched_dets=set()
        matched_s1=0; matched_s2=0; matched_s3=0

        early_locked_indices = set()
        _pkt_n_locked = 0
        _pkt_backwards_this_frame = 0  # [Phase 0] frame-local diagnostic
        _pkt_low_ncc_skipped = 0       # ensure defined even when PKT is disabled
        _pkt_soft_locked = 0
        t_pkt_ms = 0.0
        if args.use_gpu and getattr(args, 'opt_predictive_kernel', 0) == 1 and not _pkt_auto_disable:
            t_pkt0 = time.perf_counter()
            el_ids, pk_updates, _pkt_backwards_this_frame = predictive_kernel_early_lock(
                tracks, frame, previous_frame, W, H, args, threshold=0.80, search_r=10
            )
            _prof_pkt_backwards_lock_total += _pkt_backwards_this_frame

            # Build detection position array for snapping PKT locks to detection positions
            all_dets_for_snap = list(kept) + list(soft)
            if all_dets_for_snap:
                _snap_pos = np.array([[d[0], d[1]] for d in all_dets_for_snap], dtype=np.float32)
            else:
                _snap_pos = np.empty((0, 2), dtype=np.float32)
            _snap_used = set()

            n_active_tracks = sum(1 for t in tracks if t.missed == 0)
            # [Phase 4C / Phase 5] PKT two-tier NCC thresholds.
            # [Decision Rationale] 1.1x spatial exclusion (see Flowchart 7 — PKT Early Lock): slightly larger than 1.0x to account
            # for sub-pixel position jitter between NCC peak and detection center.
            # Tuned Phase 5: 1.0x caused rare double-matches; 1.2x rejected valid nearby detections.
            _pkt_spatial_mult = 1.1  # Spatial exclusion zone around PKT-locked positions

            # [Decision Rationale] NCC 0.60 (see Flowchart 7 — PKT Early Lock): below this, template correlation is unreliable
            # (background or occluded particle). These tracks must go through full Hungarian.
            _pkt_ncc_min = 0.6  # Minimum NCC for detection-snap

            mask_pts = []
            mask_radii = []
            _pkt_low_ncc_skipped = 0
            _pkt_soft_locked = 0
            # [Decision Rationale] NCC 0.85 (see Flowchart 7 — PKT Early Lock): above this, template match is unambiguous — the
            # track's position can be snapped to the NCC peak without cost matrix verification.
            # Achieves 99.7% lock rate on primary video, reducing Hungarian from 540x540 to 4x5.
            # Mar 18 tuning: 0.80 gave 99.9% lock but occasional false locks; 0.90 dropped to 98%.
            _pkt_lock_ncc_min = float(getattr(args, 'pkt_lock_thresh', 0.85))  # High-confidence threshold for full lock (Phase 3f: configurable)
            for i, t in enumerate(tracks):
                if t.id in pk_updates:
                    nx, ny, ncc_score = pk_updates[t.id]
                    # Hard skip: NCC too low for any use
                    if ncc_score < _pkt_ncc_min:
                        _pkt_low_ncc_skipped += 1
                        continue
                    # Two-tier PKT: high NCC = full lock, medium NCC = position hint only
                    _full_lock = ncc_score >= _pkt_lock_ncc_min
                    # Snap to nearest detection for positional accuracy
                    if _full_lock and len(_snap_pos) > 0:
                        dists = np.linalg.norm(_snap_pos - np.array([nx, ny], dtype=np.float32), axis=1)
                        _order = np.argsort(dists, kind="stable")
                        # [Phase 3i fix 2026-04-30] Snap radius scales
                        # with --frame-step. At step=4 the cached PKT
                        # template is stale by 4 source frames, so NCC
                        # peak biases toward old position. Without this
                        # widening, snap fails and `nx, ny` stays at the
                        # stale NCC peak -> matched-row positions drift
                        # ~2.5*step px from GT, busting F1 even though
                        # detections + linker decisions are correct.
                        _frame_step = max(1, int(getattr(args, 'frame_step', 1)))
                        _snap_radius = max(t.r * 1.5, t.r * 1.5 * _frame_step)
                        for _di in _order:
                            if dists[_di] > _snap_radius:
                                break
                            if int(_di) not in _snap_used:
                                nx, ny = float(_snap_pos[_di, 0]), float(_snap_pos[_di, 1])
                                _snap_used.add(int(_di))
                                break
                    t.px, t.py, t.pr = t.x, t.y, t.r
                    t.x, t.y = nx, ny
                    if _full_lock:
                        t.missed = 0
                        early_locked_indices.add(i)
                        matched_tracks.add(i)  # Lock it out from Hungarian
                        mask_pts.append([nx, ny])
                        mask_radii.append(t.r)
                    else:
                        # Soft lock: position updated but track goes to Hungarian for verification
                        _pkt_soft_locked += 1

            # Sanity check: if consumed detections >> active tracks, PKT is over-consuming
            _pkt_fallback = False
            if n_active_tracks > 0 and len(_snap_used) > 1.5 * n_active_tracks:
                print(f"[PKT] WARNING Frame {processed}: consumed {len(_snap_used)} dets for {n_active_tracks} tracks — falling back to Hungarian-only")
                _pkt_fallback = True
                early_locked_indices.clear()
                matched_tracks.clear()
                mask_pts.clear()
                mask_radii.clear()
                _snap_used.clear()
                # Restore track positions (undo updates)
                for i, t in enumerate(tracks):
                    if t.id in pk_updates and pk_updates[t.id][2] >= _pkt_ncc_min:
                        if hasattr(t, 'px') and t.px is not None:
                            t.x, t.y = t.px, t.py

            # Remove snapped detections + adaptive spatial filter for PKT-locked positions
            if _snap_used:
                kept_len_pre = len(kept)
                all_pre = list(kept) + list(soft)
                new_kept = []
                new_soft = []
                for idx, d in enumerate(all_pre):
                    if idx not in _snap_used:
                        if idx < kept_len_pre:
                            new_kept.append(d)
                        else:
                            new_soft.append(d)
                kept = new_kept
                soft = new_soft
            if mask_pts:
                pts_arr = np.array(mask_pts, dtype=np.float32)
                rad_arr = np.array(mask_radii, dtype=np.float32)
                _sf_mult = _pkt_spatial_mult
                def spatial_filter(dets, pts_arr=pts_arr, rad_arr=rad_arr, _sf_mult=_sf_mult):
                    """Remove detections within PKT-locked track exclusion zones."""
                    filtered = []
                    for d in dets:
                        dists = np.linalg.norm(pts_arr - np.array([d[0], d[1]]), axis=1)
                        if not np.any(dists < rad_arr * _sf_mult):
                            filtered.append(d)
                    return filtered
                kept = spatial_filter(kept)
                soft = spatial_filter(soft)

            if getattr(args, 'verbose', False):
                print(f"[PKT] Frame {processed}: Locked {len(mask_pts)} tracks ({_pkt_soft_locked} soft) via NCC ({(time.perf_counter()-t_pkt0)*1000:.1f}ms). "
                      f"Spatial={_pkt_spatial_mult:.1f}x, NCC_skip={_pkt_low_ncc_skipped}, snap_used={len(_snap_used)}")

            # Phase 6: accumulate PKT lock-rate stats
            t_pkt_ms = (time.perf_counter() - t_pkt0) * 1000.0
            _pkt_n_locked = len(mask_pts)
            _n_active = sum(1 for t in tracks if t.missed == 0)
            _prof_pkt_active_total += _n_active
            _prof_pkt_full_lock_total += len(mask_pts)
            _prof_pkt_soft_lock_total += _pkt_soft_locked
            _prof_pkt_skip_total += _pkt_low_ncc_skipped

        # [In/outflow C/D] Detections outside the visible area or inside an
        # exit region (--exit-kill) or an occluder (--occluder-coast) are not
        # particles to track: drop them before any matching or spawning.
        _io_det_dropped = 0
        if _io_on and (_io_exit or _io_occN > 0) and (kept or soft):
            def _io_keep(dets):
                if not dets:
                    return dets
                _xs = np.array([d[0] for d in dets], dtype=np.float64)
                _ys = np.array([d[1] for d in dets], dtype=np.float64)
                _lab = _io_region.labels(_xs, _ys)
                _bad = np.zeros(len(dets), dtype=bool)
                if _io_exit:
                    # T038: keep detections at / just past the frame edge (a particle resting there is still
                    # in view); drop only those beyond the --exit-margin, the same test that ends a leaving track.
                    _rs = np.array([d[2] for d in dets], dtype=np.float64)
                    _io_m_det = float(getattr(args, 'exit_margin', 1.0))
                    _bad |= (_lab == _IO_EXIT) | (_io_region.outside_dist(_xs, _ys) > _io_m_det * _rs)
                if _io_occN > 0:
                    _bad |= (_lab == _IO_OCC)
                return [d for d, b in zip(dets, _bad, strict=False) if not b]
            _n_pre = len(kept) + len(soft)
            kept = _io_keep(list(kept))
            soft = _io_keep(list(soft))
            _io_det_dropped = _n_pre - len(kept) - len(soft)

        all_candidates = list(kept) + list(soft)
        kept_len = len(kept)

        # [Phase 2] Compute candidate colours UP HERE (was in [MAIN-8]).
        # We need them in [MAIN-7.4] for the easy-match colour gate, and the
        # subsequent cost matrix in [MAIN-8] also uses them.  No re-compute.
        t_color_extract = 0.0
        t_color0 = time.perf_counter()
        colors_all = batch_compute_colors_gpu(frame, all_candidates, use_gpu=args.use_gpu)
        t_color_extract = (time.perf_counter() - t_color0) * 1000.0

        # === [MAIN-7.4] Tier-0 cascade — easy-match commit (Phase 2) ===
        # Commit unambiguous (track, candidate) pairs without a kernel
        # evaluation.  Mutates tracks / matched_tracks / matched_dets in
        # place and returns anchors for the field fit.  No-op when the
        # field flag is off.
        _n_easy_committed = 0
        easy_matches = []
        if (getattr(args, 'use_affine_field', 0) == 1
                and getattr(args, 'field_easy_commit', 1) == 1
                and tracks and all_candidates):
            # [Phase 5 Stage 3] easy-commit decoupled from the field fit; default
            # 1 preserves Phase-2/4 behaviour. 0 -> field re-centres gates on the
            # fall-through residual without the easy-pass commit liability.
            from .field import commit_easy_matches
            easy_matches, _n_easy_committed = commit_easy_matches(
                tracks, matched_tracks, matched_dets, all_candidates, colors_all, args,
                log_writer=_cluster_log_wr, frame_idx=processed,
            )

        # === [MAIN-7.5] Phase 2 — local affine deformation field ===
        # When --use-affine-field=1 we fit a per-track local affine (a, B, alpha)
        # from previous-frame velocity anchors + easy-match anchors.  The result
        # feeds two seams:
        #   • predicted_positions -> cost_matrix [COST-1/2] distance gate
        #   • affine_by_id        -> batch_build_kernels_gpu  (neighbour offset deform)
        # When flag=0, both stay None and the call sites use Phase-1 paths exactly.
        predicted_positions = None
        affine_by_id = None
        _field_alpha_sum = 0.0
        _field_alpha_pos = 0
        _field_n_anchors_sum = 0
        # [Phase 5 Stage 3.1] mu-source dispatch. Default 'field' -> only the
        # Phase-2/4 affine block below runs (byte-identical baseline). 'piv' ->
        # the anchor-free PIV field (MAIN-7.55 below) is the sole source.
        # 'piv_plus_field' -> field block is the base, PIV overrides per-track
        # where its peak is confident (alpha>0).
        _mu_source = getattr(args, 'mu_source', 'field')
        _use_field = (getattr(args, 'use_affine_field', 0) == 1) or (_mu_source == 'piv_plus_field')
        if _use_field and tracks:
            from .field import (
                compute_anchor_spread,
                deformation_confidence,
                fit_local_affine,
                predict_position,
            )
            # Collect anchors: each track with a known previous position contributes
            # one velocity sample (cur_xy, uv=cur-prev).
            _field_source = getattr(args, 'field_source', 'prev_plus_easy')
            anchor_idx_pool = []   # list of (track_idx, x, y, vx, vy)
            for _j, _tj in enumerate(tracks):
                if _tj.px is not None and _tj.py is not None:
                    anchor_idx_pool.append((_j, _tj.x, _tj.y,
                                            _tj.x - _tj.px, _tj.y - _tj.py))
            use_prev_anchors = _field_source in ("prev_plus_easy", "prev")
            use_easy_anchors = _field_source in ("prev_plus_easy", "easy")
            if (anchor_idx_pool and use_prev_anchors) or (easy_matches and use_easy_anchors):
                from scipy.spatial import cKDTree
                _anchor_xy_all = np.array(
                    [(a[1], a[2]) for a in (anchor_idx_pool if use_prev_anchors else [])] +
                    [(e[0], e[1]) for e in (easy_matches if use_easy_anchors else [])],
                    dtype=np.float64,
                )
                _anchor_uv_all = np.array(
                    [(a[3], a[4]) for a in (anchor_idx_pool if use_prev_anchors else [])] +
                    [(e[2], e[3]) for e in (easy_matches if use_easy_anchors else [])],
                    dtype=np.float64,
                )
                _anchor_track_idx = (
                    [a[0] for a in (anchor_idx_pool if use_prev_anchors else [])] +
                    [-1] * (len(easy_matches) if use_easy_anchors else 0)
                )
                # [Phase 4 Stage 2b] Temporal constellation->field propagation:
                # velocity anchors of tracks committed by a constellation in
                # the PREVIOUS frame are cluster-grade evidence (98-99%
                # audit-correct) — weight them up in the WLS fit. Flag off
                # -> weights stay None and the fit call is byte-identical.
                _anchor_w_all = None
                if getattr(args, 'field_temporal_clusters', 0) == 1 and _const_committed_prev:
                    _tw = float(getattr(args, 'field_temporal_weight', 2.5))
                    _anchor_w_all = np.ones(len(_anchor_track_idx), dtype=np.float64)
                    for _k, _ti_idx in enumerate(_anchor_track_idx):
                        if _ti_idx >= 0 and tracks[_ti_idx].id in _const_committed_prev:
                            _anchor_w_all[_k] = _tw
                _kdt = cKDTree(_anchor_xy_all)
                predicted_positions = np.array(
                    [(t.x, t.y) for t in tracks], dtype=np.float32
                )
                affine_by_id = {}
                _mean_r = float(np.mean([t.r for t in tracks]))
                _nbr_R = max(20.0, 5.0 * _mean_r)
                for _i, _ti in enumerate(tracks):
                    _idxs = _kdt.query_ball_point([_ti.x, _ti.y], r=_nbr_R)
                    # drop self if it's in there
                    _idxs = [k for k in _idxs if _anchor_track_idx[k] != _i]
                    if len(_idxs) < 3:
                        continue
                    _xy = _anchor_xy_all[_idxs]
                    _uv = _anchor_uv_all[_idxs]
                    if _anchor_w_all is not None:
                        _a, _B, _res, _n_used = fit_local_affine(
                            (_ti.x, _ti.y), _xy, _uv, weights=_anchor_w_all[_idxs])
                    else:
                        _a, _B, _res, _n_used = fit_local_affine((_ti.x, _ti.y), _xy, _uv)
                    _spread = compute_anchor_spread(_xy)
                    _alpha = deformation_confidence(_n_used, _spread, _res)
                    affine_by_id[_ti.id] = (_a, _B, float(_alpha))
                    _xp = predict_position(_a, _B, (_ti.x, _ti.y), (_ti.x, _ti.y), float(_alpha))
                    predicted_positions[_i, 0] = float(_xp[0])
                    predicted_positions[_i, 1] = float(_xp[1])
                    _field_alpha_sum += float(_alpha)
                    if _alpha > 0.0:
                        _field_alpha_pos += 1
                    _field_n_anchors_sum += _n_used

        # === [MAIN-7.55] Phase 5 Stage 3.1 — anchor-free PIV mu-source (Arm P) ===
        # Routes around the Gate-0 commit-anchor sparsity (~80% of fall-throughs
        # have no local commit field): a dense raw-pixel cross-correlation field
        # predicts (a, B, alpha) for EVERY track. previous_frame (n-1 BGR) and
        # frame (n BGR) are both in scope; no frame-retention plumbing. For
        # 'piv_plus_field' the field block above is the base and PIV overrides
        # only where confident (alpha>0); for 'piv' PIV is the sole source.
        _piv_active = (_mu_source in ('piv', 'piv_plus_field') and tracks
                       and previous_frame is not None)
        # Displacement-ceiling gate: PIV is unreliable above ~1 d motion (texture
        # decorrelation -> fragmentation). Applied PER-TRACK on each track's own
        # observed velocity/radius (a frame-level gate is too blunt: at s8 the slow
        # dense majority drags the frame mean below the gate while the fast plug
        # particles still fragment). 0 = no gate.
        _piv_gate = float(getattr(args, 'piv_max_disp_ratio', 0.0))
        _piv_pred_gate = float(getattr(args, 'piv_max_pred_ratio', 0.0))
        # [Phase 6i Part A] nd-gates motion floor: ONE-TIME m_est from the
        # first frame pair (coarse full-frame PIV median magnitude — the P6g
        # bulk-pass machinery reused as an ESTIMATOR, not a prior). Printed
        # for the per-cell gate audit table.
        # Re-estimated every 40 processed frames (raise-only: flows spin up --
        # the first-pair estimate on RES was 0.2 px vs ~8 px steady-state).
        if (getattr(args, 'nd_gates', 0) == 1 and previous_frame is not None
                and processed % 40 == 1):
            from .piv import compute_dense_piv_field as _nd_piv
            _nd_g0 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            _nd_g1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _nd_d = float(getattr(args, 'nd_dpx', 12.8))
            _nd_f = _nd_piv(_nd_g0, _nd_g1, _nd_d, args)
            _nd_m = np.hypot(_nd_f.u, _nd_f.v)
            # CONFIDENT windows only: textureless/empty regions return junk
            # vectors at sigma == max_sigma (caught by the parity audit:
            # m_est 41 px on an ~8 px bed) -- exclude them from the median.
            _nd_conf = _nd_f.sigma < (float(getattr(args, 'piv_max_sigma', 8.0))
                                      - 1e-6)
            _nd_m = _nd_m[_nd_conf]
            _nd_mest = float(np.median(_nd_m)) if _nd_m.size else 0.0
            _nd_floor = float(getattr(args, 'nd_km', 1.25)) * _nd_mest
            if _nd_floor > args.max_dist:
                print(f"[ND-GATES] motion floor ENGAGED at f{processed}: "
                      f"m_est={_nd_mest:.1f}px -> max_dist {args.max_dist} -> "
                      f"{round(_nd_floor, 2)}", flush=True)
                args.max_dist = round(_nd_floor, 2)
            elif processed == 1:
                print(f"[ND-GATES] motion floor inert at f1: "
                      f"m_est={_nd_mest:.1f}px (floor {round(_nd_floor, 2)} "
                      f"<= max_dist {args.max_dist})", flush=True)

        _piv_bulk = None
        _coh_route = {}
        if _piv_active:
            from .field import predict_position as _predict_position
            from .piv import compute_dense_piv_field, sample_piv_affine_at_particle
            _g_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            _g_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _piv_mean_r = float(np.mean([t.r for t in tracks]))
            _piv_d = max(2.0 * _piv_mean_r, 2.0)
            # [Phase 6g Stage 3.2] --piv-bulk-pass: estimate the whole-frame
            # bulk shift once, re-centre the PIV windows by it, and gate on
            # the RESIDUAL velocity below. Default 0 -> _piv_bulk stays None
            # and every downstream expression is the pre-6g path.
            if getattr(args, 'piv_bulk_pass', 0) == 1:
                from .piv import estimate_bulk_shift
                _piv_bulk = estimate_bulk_shift(_g_prev, _g_curr)
            # [Phase 6h] under the outlier gate the dense field is consumed
            # ONLY by the optional override; skip the ~0.5-1 s/frame compute
            # when the override is off. Velocity/coherence paths unchanged.
            _out_active = (_mu_source == 'piv'
                           and getattr(args, 'piv_gate', 'velocity') == 'outlier')
            _out_override = getattr(args, 'outlier_override', 0) == 1
            if _out_active and not _out_override:
                _piv = None
            else:
                _piv = compute_dense_piv_field(_g_prev, _g_curr, _piv_d, args,
                                               use_gpu=bool(getattr(args, 'use_gpu', 0)),
                                               bulk=_piv_bulk)
            _piv_bs = float(getattr(args, 'piv_base_sigma', 2.0))
            _piv_ms = float(getattr(args, 'piv_max_sigma', 8.0))
            if predicted_positions is None:
                predicted_positions = np.array([(t.x, t.y) for t in tracks], dtype=np.float32)
            if affine_by_id is None:
                affine_by_id = {}
            # === [Phase 6g rev 3, arm 3] coherence-routed PIV gate ===
            # Self-contained branch; the shipped velocity-gate loop below is
            # untouched (byte-identical when --piv-gate velocity, the default).
            _coh_active = (_mu_source == 'piv'
                           and getattr(args, 'piv_gate', 'velocity') == 'coherence')
            if _coh_active:
                from scipy.spatial import cKDTree as _CohTree
                _coh_conf_min = float(getattr(args, 'piv_conf_min', 0.3))
                _coh_disp_max = float(getattr(args, 'nvel_disp_max', 0.02))
                _coh_k = max(2, int(getattr(args, 'nvel_k', 8)))
                # wrap-cleaned live velocity pool (P6f T_gran convention:
                # drop speeds > 5x the median -- recycle/teleport guard)
                _cv_rows = [(_i, _t.x, _t.y, _t.x - _t.px, _t.y - _t.py)
                            for _i, _t in enumerate(tracks)
                            if _t.px is not None and _t.py is not None]
                _cv_tree = None
                if _cv_rows:
                    _cv_uv = np.array([(r[3], r[4]) for r in _cv_rows])
                    _cv_speed = np.hypot(_cv_uv[:, 0], _cv_uv[:, 1])
                    _cv_med = float(np.median(_cv_speed))
                    _cv_keep = (_cv_speed <= 5.0 * _cv_med) if _cv_med > 0 \
                        else np.ones(len(_cv_rows), bool)
                    _cv_rows = [r for r, k in zip(_cv_rows, _cv_keep, strict=False) if k]
                    if _cv_rows:
                        _cv_xy = np.array([(r[1], r[2]) for r in _cv_rows])
                        _cv_uv = np.array([(r[3], r[4]) for r in _cv_rows])
                        _cv_idx = np.array([r[0] for r in _cv_rows])
                        _cv_tree = _CohTree(_cv_xy)
                _coh_B0 = np.zeros((2, 2), dtype=np.float64)
                _coh_d2 = float(_piv_d) ** 2
                for _pi, _pt in enumerate(tracks):
                    _pa, _pB, _pal = sample_piv_affine_at_particle(
                        _piv, _pt.x, _pt.y, _piv_d,
                        base_sigma=_piv_bs, max_sigma=_piv_ms)
                    if _pal >= _coh_conf_min:
                        # sharp peak -> trust PIV (keeps the collision cells
                        # PIV-routed: pixels are right even when flow is not)
                        affine_by_id[_pt.id] = (_pa, _pB, float(_pal))
                        _pxp = _predict_position(_pa, _pB, (_pt.x, _pt.y),
                                                 (_pt.x, _pt.y), float(_pal))
                        predicted_positions[_pi, 0] = float(_pxp[0])
                        predicted_positions[_pi, 1] = float(_pxp[1])
                        _field_alpha_sum += float(_pal)
                        _field_alpha_pos += 1
                        _coh_route[_pt.id] = 'coh-piv'
                        continue
                    # low PIV confidence: velocity extrapolation only where the
                    # local velocity field is coherent (low dispersion)
                    _disp_ok = False
                    _u_med = None
                    if _cv_tree is not None and len(_cv_idx) >= 3:
                        _kq = min(_coh_k + 1, len(_cv_idx))
                        _, _qi = _cv_tree.query([_pt.x, _pt.y], k=_kq)
                        _qi = [int(q) for q in np.atleast_1d(_qi)
                               if int(_cv_idx[q]) != _pi][:_coh_k]
                        if len(_qi) >= 3:
                            _nuv = _cv_uv[_qi]
                            _u_med = np.median(_nuv, axis=0)
                            _fl = _nuv - _u_med
                            _dispv = float((_fl ** 2).sum(axis=1).mean()) / _coh_d2
                            _disp_ok = _dispv <= _coh_disp_max
                    if _disp_ok:
                        if _pt.px is not None and _pt.py is not None:
                            _nv = np.array([_pt.x - _pt.px, _pt.y - _pt.py],
                                           dtype=np.float64)
                            _coh_route[_pt.id] = 'coh-ownvel'
                        else:
                            _nv = np.asarray(_u_med, dtype=np.float64)
                            _coh_route[_pt.id] = 'coh-nbrvel'
                        affine_by_id[_pt.id] = (_nv, _coh_B0, 1.0)
                        predicted_positions[_pi, 0] = float(_pt.x + _nv[0])
                        predicted_positions[_pi, 1] = float(_pt.y + _nv[1])
                        _field_alpha_sum += 1.0
                        _field_alpha_pos += 1
                    else:
                        # both signals bad -> static (no prior; identity gate)
                        _coh_route[_pt.id] = 'coh-static'

            # === [Phase 6h Part B] validated-extrapolation (outlier) gate ===
            # Certify the EXTRAPOLATOR: own velocity validated against the
            # k-NN neighbour median (absolute tolerance in diameters, B0-
            # calibrated); birth tracks take a coherent neighbourhood's
            # median; a sharp PIV peak that DISAGREES with the prior may
            # override it (the impact detector, --outlier-override 1).
            # Self-contained; velocity/coherence paths untouched.
            if _out_active and tracks:
                from scipy.spatial import cKDTree as _OutTree
                _out_k = max(2, int(getattr(args, 'outlier_k', 5)))
                _out_tol = float(getattr(args, 'outlier_tol_d', 0.2)) * _piv_d
                _out_conf = float(getattr(args, 'piv_conf_hi', 0.9))
                _out_marg = float(getattr(args, 'override_margin', 0.5)) * _piv_d
                _ov_rows = [(_i, _t.x, _t.y, _t.x - _t.px, _t.y - _t.py)
                            for _i, _t in enumerate(tracks)
                            if _t.px is not None and _t.py is not None]
                _ov_tree = None
                if _ov_rows:
                    _ov_xy = np.array([(r[1], r[2]) for r in _ov_rows])
                    _ov_uv = np.array([(r[3], r[4]) for r in _ov_rows])
                    _ov_idx = np.array([r[0] for r in _ov_rows])
                    _ov_tree = _OutTree(_ov_xy)
                _out_B0 = np.zeros((2, 2), dtype=np.float64)
                for _pi, _pt in enumerate(tracks):
                    _prior_v = None
                    _route = 'out-static'
                    if _ov_tree is not None and len(_ov_idx) >= 4:
                        _kq = min(_out_k + 1, len(_ov_idx))
                        _, _qi = _ov_tree.query([_pt.x, _pt.y], k=_kq)
                        _nbr = [int(q) for q in np.atleast_1d(_qi)
                                if int(_ov_idx[q]) != _pi][:_out_k]
                        if len(_nbr) >= 3:
                            _med = np.median(_ov_uv[_nbr], axis=0)
                            if _pt.px is not None and _pt.py is not None:
                                # validation test: own velocity vs the median
                                _vown = np.array([_pt.x - _pt.px,
                                                  _pt.y - _pt.py])
                                if np.hypot(*(_vown - _med)) <= _out_tol:
                                    _prior_v = _vown
                                    _route = 'out-ownvel'
                            else:
                                # birth track: coherent neighbourhood's median
                                _mad = float(np.median(
                                    [np.hypot(*(_ov_uv[j] - _med))
                                     for j in _nbr]))
                                if _mad <= _out_tol:
                                    _prior_v = np.asarray(_med, dtype=np.float64)
                                    _route = 'out-nbrvel'
                    if _out_override and _piv is not None:
                        # sharp pixels beat agreed-upon history
                        _pa, _pB, _pal = sample_piv_affine_at_particle(
                            _piv, _pt.x, _pt.y, _piv_d,
                            base_sigma=_piv_bs, max_sigma=_piv_ms)
                        _pv = _prior_v if _prior_v is not None \
                            else np.zeros(2, dtype=np.float64)
                        if (_pal >= _out_conf
                                and np.hypot(_pa[0] - _pv[0],
                                             _pa[1] - _pv[1]) > _out_marg):
                            affine_by_id[_pt.id] = (_pa, _pB, float(_pal))
                            _pxp = _predict_position(
                                _pa, _pB, (_pt.x, _pt.y), (_pt.x, _pt.y),
                                float(_pal))
                            predicted_positions[_pi, 0] = float(_pxp[0])
                            predicted_positions[_pi, 1] = float(_pxp[1])
                            _field_alpha_sum += float(_pal)
                            _field_alpha_pos += 1
                            _coh_route[_pt.id] = 'out-override'
                            continue
                    if _prior_v is not None:
                        affine_by_id[_pt.id] = (_prior_v, _out_B0, 1.0)
                        predicted_positions[_pi, 0] = float(_pt.x + _prior_v[0])
                        predicted_positions[_pi, 1] = float(_pt.y + _prior_v[1])
                        _field_alpha_sum += 1.0
                        _field_alpha_pos += 1
                    _coh_route[_pt.id] = _route

            for _pi, _pt in enumerate(tracks) if not (_coh_active or _out_active) else ():
                # Per-track displacement-ceiling gate: skip PIV for fast tracks
                # (own motion > gate*radius) where the local texture has
                # decorrelated; they keep the field/none prediction.
                # [Phase 6g] Under --piv-bulk-pass the gate applies to the
                # RESIDUAL after bulk removal: a fast-but-coherent bed no
                # longer disengages the prior.
                if _piv_gate > 0.0 and _pt.px is not None and _pt.py is not None:
                    if _piv_bulk is None:
                        _piv_vel = np.hypot(_pt.x - _pt.px, _pt.y - _pt.py)
                    else:
                        _piv_vel = np.hypot(_pt.x - _pt.px - _piv_bulk[0],
                                            _pt.y - _pt.py - _piv_bulk[1])
                    if _piv_vel > _piv_gate * _piv_mean_r:
                        continue
                _pa, _pB, _pal = sample_piv_affine_at_particle(
                    _piv, _pt.x, _pt.y, _piv_d, base_sigma=_piv_bs, max_sigma=_piv_ms)
                if _mu_source == 'piv_plus_field' and _pal <= 0.0:
                    continue  # keep the field's prediction where PIV is flat
                _pxp = _predict_position(_pa, _pB, (_pt.x, _pt.y), (_pt.x, _pt.y), float(_pal))
                # [Phase 5 gs_disp] Prediction-magnitude ceiling: a PIV vector
                # larger than the decorrelation ceiling is untrustworthy no
                # matter the track's own velocity — at high displacement the
                # moving texture contaminates the windows of NEAR-STATIC
                # neighbours (velocity ~0, so --piv-max-disp-ratio can't catch
                # them). Skip the prediction; the track keeps its velocity/none
                # prior. Default 0 = off = byte-identical.
                if (_piv_pred_gate > 0.0
                        and np.hypot(_pxp[0] - _pt.x, _pxp[1] - _pt.y)
                            > _piv_pred_gate * _piv_mean_r):
                    continue
                affine_by_id[_pt.id] = (_pa, _pB, float(_pal))
                predicted_positions[_pi, 0] = float(_pxp[0])
                predicted_positions[_pi, 1] = float(_pxp[1])
                _field_alpha_sum += float(_pal)
                if _pal > 0.0:
                    _field_alpha_pos += 1

        # === [MAIN-7.56] Phase 6c — ordered-flow NVP mu-source ===
        # trackpy-NearestVelocityPredict-style prior for coherent plug flow
        # (P6 finding: constant-velocity extrapolation is ~exact there and
        # beats the PIV window correlation). mu = the track's OWN last
        # velocity when known; a track without one (birth / post-occlusion)
        # borrows the median velocity of its --nvp-k nearest tracks that
        # have one. Pure translation: B = 0 and alpha = 1, so the kernel
        # build is byte-identical and only the gate re-centring moves. Never
        # entered unless --mu-source nvp (default 'field' untouched).
        if _mu_source == 'nvp' and tracks:
            from scipy.spatial import cKDTree as _NvpTree
            _vel_rows = [(_t.x, _t.y, _t.x - _t.px, _t.y - _t.py)
                         for _t in tracks if _t.px is not None and _t.py is not None]
            if predicted_positions is None:
                predicted_positions = np.array([(t.x, t.y) for t in tracks], dtype=np.float32)
            if affine_by_id is None:
                affine_by_id = {}
            if _vel_rows:
                _nvp_xy = np.array([(s[0], s[1]) for s in _vel_rows], dtype=np.float64)
                _nvp_uv = np.array([(s[2], s[3]) for s in _vel_rows], dtype=np.float64)
                _nvp_tree = _NvpTree(_nvp_xy)
                _nvp_k = max(1, int(getattr(args, 'nvp_k', 1)))
                _nvp_B0 = np.zeros((2, 2), dtype=np.float64)
                for _ni, _nt in enumerate(tracks):
                    if _nt.px is not None and _nt.py is not None:
                        _nv = np.array([_nt.x - _nt.px, _nt.y - _nt.py], dtype=np.float64)
                    else:
                        _kq = min(_nvp_k, len(_vel_rows))
                        _ndist, _nidx = _nvp_tree.query([_nt.x, _nt.y], k=_kq)
                        _nv = np.median(_nvp_uv[np.atleast_1d(_nidx)], axis=0)
                    affine_by_id[_nt.id] = (_nv, _nvp_B0, 1.0)
                    predicted_positions[_ni, 0] = float(_nt.x + _nv[0])
                    predicted_positions[_ni, 1] = float(_nt.y + _nv[1])
                    _field_alpha_sum += 1.0
                    _field_alpha_pos += 1

        # === [MAIN-7.57] Phase 6g — pure neighbour-field mu-source ===
        # Every track's mu = component-wise MEDIAN velocity of its --nfield-k
        # nearest OTHER tracks with known velocity. Never own-velocity-first
        # (that was nvp's stale-feedback path): one wrong link corrupts
        # only 1/k of each neighbour's prior instead of the track's own whole
        # prior. New tracks (px None) get the field immediately (no birth
        # gap). Pure translation: B = 0, alpha = 1 (gate re-centring only).
        # Never entered unless --mu-source nfield (default 'field' untouched).
        if _mu_source == 'nfield' and tracks:
            from scipy.spatial import cKDTree as _NfTree
            _nf_src = [(_i, _t.x, _t.y, _t.x - _t.px, _t.y - _t.py)
                       for _i, _t in enumerate(tracks)
                       if _t.px is not None and _t.py is not None]
            if predicted_positions is None:
                predicted_positions = np.array([(t.x, t.y) for t in tracks], dtype=np.float32)
            if affine_by_id is None:
                affine_by_id = {}
            if _nf_src:
                _nf_idx = np.array([s[0] for s in _nf_src], dtype=np.int64)
                _nf_xy = np.array([(s[1], s[2]) for s in _nf_src], dtype=np.float64)
                _nf_uv = np.array([(s[3], s[4]) for s in _nf_src], dtype=np.float64)
                _nf_tree = _NfTree(_nf_xy)
                _nf_k = max(1, int(getattr(args, 'nfield_k', 5)))
                _nf_B0 = np.zeros((2, 2), dtype=np.float64)
                for _fi, _ft in enumerate(tracks):
                    _kq = min(_nf_k + 1, len(_nf_src))
                    _, _qi = _nf_tree.query([_ft.x, _ft.y], k=_kq)
                    _qi = np.atleast_1d(_qi)
                    _sel = [int(q) for q in _qi if int(_nf_idx[q]) != _fi][:_nf_k]
                    if not _sel:
                        continue
                    _nv = np.median(_nf_uv[_sel], axis=0)
                    affine_by_id[_ft.id] = (_nv, _nf_B0, 1.0)
                    predicted_positions[_fi, 0] = float(_ft.x + _nv[0])
                    predicted_positions[_fi, 1] = float(_ft.y + _nv[1])
                    _field_alpha_sum += 1.0
                    _field_alpha_pos += 1

        # === [In/outflow C/D] pre-matching exit + occluder bookkeeping ===
        # Predictions are final here. A track whose predicted position lies
        # in an exit region, or more than --exit-margin radii outside the
        # visible area, has left: it is terminated THIS frame and kept out of
        # every matching stage (constellations, cost rows, second pass,
        # recovery) so it cannot claim an entering particle. It is NOT added
        # to matched_tracks (that would reset its miss counter). Tracks that
        # coasted through an occluder are gated tightly around their velocity
        # prediction and kept out of the auxiliary matchers.
        # Tracks already matched upstream (PKT, easy-commit) are left alone.
        _io_excl = set()
        _io_occl = set()
        _io_skip = None
        if _io_on and tracks:
            _io_n = len(tracks)
            _io_px = np.empty(_io_n, dtype=np.float64)
            _io_py = np.empty(_io_n, dtype=np.float64)
            for _i, _t in enumerate(tracks):
                if (predicted_positions is not None and affine_by_id
                        and _t.id in affine_by_id):
                    _io_px[_i] = float(predicted_positions[_i, 0])
                    _io_py[_i] = float(predicted_positions[_i, 1])
                elif _io_coast != 'freeze':
                    _io_px[_i], _io_py[_i] = _t.x + _t.vx, _t.y + _t.vy
                else:
                    _io_px[_i], _io_py[_i] = _t.x, _t.y
            if _io_occN > 0:
                for _i, _t in enumerate(tracks):
                    if _t.occl_n > 0 and _i not in matched_tracks:
                        _io_occl.add(_i)
                        _io_px[_i], _io_py[_i] = _t.x + _t.vx, _t.y + _t.vy
                if _io_occl:
                    if predicted_positions is None:
                        predicted_positions = np.array([(t.x, t.y) for t in tracks],
                                                       dtype=np.float32)
                    for _i in _io_occl:
                        predicted_positions[_i, 0] = _io_px[_i]
                        predicted_positions[_i, 1] = _io_py[_i]
            if _io_exit:
                _io_lab = _io_region.labels(_io_px, _io_py)
                _io_out = _io_region.outside_dist(_io_px, _io_py)
                _io_m = float(getattr(args, 'exit_margin', 1.0))
                # T038: also judge by the track's own velocity, so a particle moving out is caught even when
                # the motion prior (e.g. a static flow field) predicts it standing still.
                _io_vout = _io_region.outside_dist(np.array([t.x + t.vx for t in tracks], dtype=np.float64),
                                                   np.array([t.y + t.vy for t in tracks], dtype=np.float64))
                _io_last_out = _io_region.outside_dist(np.array([t.x for t in tracks], dtype=np.float64),
                                                       np.array([t.y for t in tracks], dtype=np.float64))
                for _i, _t in enumerate(tracks):
                    if _i in matched_tracks:
                        continue
                    if _io_lab[_i] == _IO_EXIT:          # exit regions (label 1): unchanged
                        _t.death_reason = 'exit_region'
                    # T038, frame edge only: just LEAVING tracks are ended before matching, i.e. a coasting
                    # track or one whose last matched position was inside the visible area; a track matched
                    # last frame at / past the edge (a particle resting there) stays in matching.
                    elif ((_t.missed > 0 or _io_last_out[_i] == 0.0)
                          and max(_io_out[_i], _io_vout[_i]) > _io_m * _t.r):
                        _t.death_reason = 'exit_frame'
                    else:
                        continue
                    _io_excl.add(_i)
                    _io_occl.discard(_i)
            _io_skip = (_io_excl | _io_occl) or None

        # [Phase 6g Stage 1] linkdump snapshot: per-track prior source +
        # predicted position, taken at the moment predictions are consumed.
        # Pure reads; runs only when --debug-linkdump is set.
        _ld_pre = None
        _ld_stage = {}
        if _linkdump_wr is not None and tracks:
            _ld_pre = []
            # per-gate candidate counts + same-colour fractions (Stage 3.0
            # lattice-ambiguity read): distances from each track's PREDICTED
            # position to every candidate, against args.max_dist.
            if all_candidates:
                _ld_cand_xy = np.array([(c[0], c[1]) for c in all_candidates],
                                       dtype=np.float64)
                _ld_cand_dom = np.argmax(np.asarray(colors_all, dtype=np.float64),
                                         axis=1)
            else:
                _ld_cand_xy = None
                _ld_cand_dom = None
            for _li, _lt in enumerate(tracks):
                if predicted_positions is not None:
                    _lpx = float(predicted_positions[_li, 0])
                    _lpy = float(predicted_positions[_li, 1])
                else:
                    _lpx, _lpy = float(_lt.x), float(_lt.y)
                _lent = affine_by_id.get(_lt.id) if affine_by_id else None
                _lalpha = float(_lent[2]) if _lent is not None else -1.0
                _lvel = _lt.px is not None and _lt.py is not None
                if _coh_route:
                    # [P6g rev 3] coherence-routed frame: the route label IS
                    # the prior source (coh-piv/coh-ownvel/coh-nbrvel/coh-static)
                    _lsrc = _coh_route.get(_lt.id, 'coh-?')
                elif _mu_source in ('piv', 'piv_plus_field'):
                    if _lent is not None:
                        _lsrc = 'piv'
                    elif not _piv_active:
                        _lsrc = 'none'
                    else:
                        if _piv_gate > 0.0 and _lvel:
                            if _piv_bulk is None:
                                _lv = np.hypot(_lt.x - _lt.px, _lt.y - _lt.py)
                            else:
                                _lv = np.hypot(_lt.x - _lt.px - _piv_bulk[0],
                                               _lt.y - _lt.py - _piv_bulk[1])
                            _lsrc = ('piv-gated-fast' if _lv > _piv_gate * _piv_mean_r
                                     else 'piv-nopred')
                        else:
                            _lsrc = 'piv-nopred'
                elif _mu_source == 'nvp':
                    _lsrc = ('own-vel' if _lvel else 'nbr-vel') if _lent is not None else 'none'
                elif _mu_source == 'nfield':
                    _lsrc = 'nfield' if _lent is not None else 'nfield-none'
                elif _mu_source == 'none':
                    _lsrc = 'none'
                else:
                    _lsrc = 'field' if _lent is not None else 'none'
                if _ld_cand_xy is not None:
                    _ld_d = np.hypot(_ld_cand_xy[:, 0] - _lpx,
                                     _ld_cand_xy[:, 1] - _lpy)
                    _ld_in = _ld_d <= float(args.max_dist)
                    _ld_ng = int(_ld_in.sum())
                    _ld_tdom = int(np.argmax(np.asarray(_lt.color_bgr,
                                                        dtype=np.float64)))
                    _ld_ns = int((_ld_cand_dom[_ld_in] == _ld_tdom).sum())
                else:
                    _ld_ng = _ld_ns = 0
                _ld_pre.append((_lt.id, float(_lt.x), float(_lt.y), _lpx, _lpy,
                                _lsrc, _lalpha, int(_lvel), int(_lt.missed),
                                _ld_ng, _ld_ns))
            for _li in matched_tracks:
                _ld_stage[_li] = 'pre'
            _ld_pre_matched = set(matched_tracks)

        # === [MAIN-7.6] Phase 3 — colour-constellation joint assignment ===
        # Frame order: easy -> field -> constellations -> residual hard set
        # through the normal cost matrix. Committed members and their
        # targets enter matched_tracks/matched_dets, so they leave the
        # assignment stages exactly like easy matches do (smaller
        # Hungarian). All cluster logic lives in cbtv.clusters.
        _const_stats = {"proposed": 0, "accepted": 0, "members_committed": 0,
                        "outliers": 0, "field_refreshed": 0}
        t_const = 0.0
        if getattr(args, 'use_constellations', 0) == 1 and tracks and all_candidates:
            from .clusters import run_constellations
            # [Phase 6h Part A] per-N cluster cap: scale the proposal cap with
            # the frame's detection count so large-N cells are not stranded in
            # the residual matcher (the P6g coverage mechanism at scale).
            # Flag off (default) -> args.cluster_max_per_frame untouched.
            if getattr(args, 'cluster_cap_auto', 0) == 1:
                if not hasattr(args, '_ccap_base'):
                    args._ccap_base = int(getattr(args, 'cluster_max_per_frame', 128))
                args.cluster_max_per_frame = min(
                    int(getattr(args, 'cluster_cap_max', 2048)),
                    max(args._ccap_base,
                        int(math.ceil(float(getattr(args, 'cluster_cap_alpha', 0.324))
                                      * len(all_candidates)))))
            _t_const0 = time.perf_counter()
            _const_stats = run_constellations(
                tracks, matched_tracks, matched_dets, all_candidates, colors_all,
                affine_by_id, predicted_positions, args,
                frame_idx=processed, log_writer=_cluster_log_wr,
                excluded=_io_skip,
            )
            t_const = (time.perf_counter() - _t_const0) * 1000.0
        # [Phase 4 Stage 2b] Remember this frame's constellation commits for
        # the next frame's field fit (empty when constellations are off).
        _const_committed_prev = set(_const_stats.get("committed_track_ids", []))
        if _linkdump_wr is not None and _ld_pre is not None:
            for _li in matched_tracks - _ld_pre_matched:
                _ld_stage[_li] = 'const'

        # === [MAIN-8] Cost matrix construction (see Flowcharts 8, 13, 15) ===
        # GPU-batched colour extraction, neighbour lists
        # (precompute_neighbor_lists), GPU kernel build
        # (batch_build_kernels_gpu), and compute_cost_matrix_gpu —
        # which calls compute_adaptive_weights_normalized internally —
        # produce the M x N cost matrix C fed to the assignment stages.
        # Initialize GPU profiling metrics
        # (colors_all + t_color_extract were computed above for [MAIN-7.4].)
        kernel_info = {'t_dist': 0.0, 't_color': 0.0, 't_kernel_build': 0.0,
                       't_kernel_corr': 0.0, 'kernel_hits': 0, 'kernel_misses': 0}

        _hungarian_rows = 0
        _hungarian_cols = 0
        if tracks and all_candidates:
            # GPU-ACCELERATED: Vectorized cost matrix computation
            tracks_soa = TrackArrays(tracks)

            # PRE-BUILD KERNELS IN PARALLEL (once per frame, not per track-detection pair)
            # This reduces kernel building from O(M×N) to O(M) with multithreading
            kernel_cache.set_frame(processed)

            # [Phase 3 Stage B] Displacement-adaptive cost weights: when
            # --adaptive-cost-weights 1, the observed mean easy-match
            # displacement / mean radius drives the effective kernel
            # weight (0..--w-kernel) and the distance-gate width for THIS
            # frame's cost matrix only. _args_cm is `args` itself when the
            # flag is off (byte-identical path).
            _args_cm = args
            _w_kernel_eff = getattr(args, 'w_kernel', 0.0)
            if getattr(args, 'adaptive_cost_weights', 0) == 1:
                from .field import adaptive_args_view
                _args_cm, _w_kernel_eff, _adp_ratio = adaptive_args_view(
                    args, easy_matches, tracks)

            # [Phase 3 Stage A] Kernel build gated on w_kernel > 0: with the
            # kernel-NCC term off (w_kernel = 0, the default and every
            # historical run), kernels were built every frame and never
            # consumed by the cost matrix — pure wasted wall-clock. The
            # cost-matrix branch that reads kernel_cache is dead unless
            # w_dist + w_color < 1.0, which (post-validate_args) is exactly
            # w_kernel > 0, so gating the build on the same condition
            # cannot change any score. batch_kernel_recovery builds its own
            # kernels and does not depend on this prebuild.
            _neighbor_counts = None
            graph_struct_data = None
            _use_gs = getattr(args, 'match_op', 'kernel') == 'graph-struct'
            if (args.use_neighborhood_kernels and _w_kernel_eff > 0.0
                    and previous_frame is not None and previous_detections):
                # Standard kernel building (no complex manager needed)
                    # Step 1: Pre-compute neighbor lists with FIXED sizing (no adaptive overhead)
                    t_neigh0 = time.perf_counter()
                    neighbor_lists, neighbor_data = precompute_neighbor_lists(
                        tracks_soa, previous_detections, args.neighbor_search_factor, adaptive_sizing=False
                    )
                    # [Phase 3 Stage A] Per-track neighbour counts for the
                    # adaptive-weight neigh_factor (wart fix: the vectorized
                    # cost path used the constant args.min_neighbors here).
                    # Same count that gates the kernel build itself.
                    _neighbor_counts = np.array(
                        [len(neighbor_lists.get(int(tid), [])) for tid in tracks_soa.ids],
                        dtype=np.int32,
                    )

                    if _use_gs:
                        # [Phase 5 Stage 3] --match-op graph-struct: SKIP the LAB
                        # kernel BUILD (the ~320 ms hotspot). Build the per-track
                        # neighbour star-graph data instead (offsets rel to the
                        # track's prev position + previous-frame colours + the
                        # local PIV transform); cost_matrix substitutes the
                        # kernel-free structural score for the NCC term.
                        from .match_ops import build_graph_struct_data
                        _prev_xy = np.array([[d[0], d[1]] for d in previous_detections], dtype=np.float64)
                        _prev_col = np.asarray(
                            batch_compute_colors_gpu(previous_frame, previous_detections, use_gpu=args.use_gpu),
                            dtype=np.float64)
                        graph_struct_data = build_graph_struct_data(
                            np.asarray(tracks_soa.positions, dtype=np.float64), tracks_soa.ids,
                            neighbor_lists, _prev_xy, _prev_col, affine_by_id, kmax=8)
                        prebuild_time = 0.0
                        prebuild_count = 0
                    else:
                        # Step 2: Build kernels with adaptive sizing.
                        # [Phase 4 Stage 1, finding B] Default: build ONLY for
                        # cache-miss tracks — the put loop below is
                        # put-if-absent, so builds for cached tracks were
                        # always discarded; skipping them is byte-identical
                        # (same cache evolution, same NCC inputs) and cuts
                        # ~5/6 of the build cost. --kernel-cache-fresh 1
                        # builds + stores everything (age-0 kernels).
                        _cache_fresh = getattr(args, 'kernel_cache_fresh', 0) == 1
                        if args.opt_gpu_kernel == 1 and args.use_gpu:
                            if _cache_fresh:
                                _build_ids = None
                            else:
                                _build_ids = {int(tid) for tid in tracks_soa.ids
                                              if kernel_cache.get(int(tid)) is None}
                            # [Phase 4 Stage 3] --kernel-deform-off science cell:
                            # build kernels undeformed (alpha=0) while the field
                            # still drives gates/constellations.
                            _kb_affine = None if getattr(args, 'kernel_deform_off', 0) == 1 else affine_by_id
                            prebuilt_kernels = batch_build_kernels_gpu(
                                tracks_soa, previous_detections, previous_frame, prev_frame_lab_gpu,
                                neighbor_lists, args, use_gpu=True, neighbor_data=neighbor_data,
                                affine_by_id=_kb_affine, only_track_ids=_build_ids,
                            )
                            build_timing = {'t_kernel_build': (time.perf_counter() - t_neigh0) * 1000.0, 'kernel_count': len(prebuilt_kernels)}
                        else:
                            prebuilt_kernels, build_timing = prebuild_all_kernels_parallel(
                                tracks_soa, previous_detections, previous_frame,
                                neighbor_lists, args, max_workers=None, neighbor_data=neighbor_data,
                                prev_lab=prev_frame_lab_gpu, kernel_cache=kernel_cache
                            )

                        # Step 3: Store newly built kernels in cache for O(1) lookup during cost matrix computation
                        # [Phase 3 Stage B] --kernel-cache-fresh 1 stores every
                        # fresh build (kernels age 0 instead of 0-5 frames);
                        # default 0 = shipped put-if-absent (byte-identical).
                        for track_id, kernel_data in prebuilt_kernels.items():
                            if _cache_fresh or kernel_cache.get(track_id) is None: # Only put if it was just built
                                # Find track position to store with kernel
                                t_idx = list(tracks_soa.ids).index(track_id)
                                t_pos = (float(tracks_soa.positions[t_idx, 0]), float(tracks_soa.positions[t_idx, 1]))
                                kernel_cache.put(track_id, kernel_data, t_pos)

                        # Store pre-build timing for later merge
                        prebuild_time = build_timing['t_kernel_build']
                        prebuild_count = build_timing['kernel_count']
            else:
                prebuild_time = 0.0
                prebuild_count = 0  # noqa: F841 -- kept: mirrors the build branch's prebuild_count

            # Compute cost matrix with optional simple spatial filtering
            if args.use_adaptive_kernel_search:
                # ULTRA-SIMPLE SPATIAL FILTER: Just KD-tree, no complex preprocessing!
                import time as time_spatial

                from scipy.spatial import cKDTree

                t0_spatial = time_spatial.perf_counter()

                # Build KD-tree from detections
                det_positions = np.array([[x, y] for x, y, r, s in all_candidates])
                kdtree = cKDTree(det_positions)

                # Find nearby pairs within search radius
                # Use second_pass_factor to ensure we don't reject valid second-pass pairs
                search_radius = args.max_dist * args.second_pass_factor
                nearby_pairs = []
                for i, track in enumerate(tracks):
                    nearby_indices = kdtree.query_ball_point([track.x, track.y], r=search_radius)
                    for j in nearby_indices:
                        nearby_pairs.append((i, j))

                t_spatial = (time_spatial.perf_counter() - t0_spatial) * 1000.0

                # Standard GPU cost computation
                C, kernel_info_corr = compute_cost_matrix_gpu(
                    tracks_soa, all_candidates, colors_all, frame, frame_lab,
                    previous_frame, previous_detections, kernel_cache, _args_cm,
                    use_gpu=args.use_gpu, frame_lab_gpu=frame_lab_gpu,
                    predicted_positions=predicted_positions,
                    neighbor_counts=_neighbor_counts,
                    graph_struct_data=graph_struct_data,
                    dump_frame=processed,
                )

                # Apply spatial filter: set non-nearby pairs to very high cost
                mask = np.ones((len(tracks), len(all_candidates)), dtype=bool)
                for i, j in nearby_pairs:
                    mask[i, j] = False
                C[mask] = 1e6

                if args.verbose:
                    accepted = len(nearby_pairs)
                    total = len(tracks) * len(all_candidates)
                    print(f"[SPATIAL FILTER] KD-tree: {t_spatial:.1f}ms, kept {accepted}/{total} pairs ({100*accepted/total:.1f}%), rejected {total-accepted}")
            else:
                # Standard cost matrix computation (no filtering)
                C, kernel_info_corr = compute_cost_matrix_gpu(
                    tracks_soa, all_candidates, colors_all, frame, frame_lab,
                    previous_frame, previous_detections, kernel_cache, _args_cm,
                    use_gpu=args.use_gpu, frame_lab_gpu=frame_lab_gpu,
                    predicted_positions=predicted_positions,
                    neighbor_counts=_neighbor_counts,
                    graph_struct_data=graph_struct_data,
                    dump_frame=processed,
                )

            # Merge kernel info: pre-build timing + correlation timing
            kernel_info['t_dist'] = kernel_info_corr['t_dist']
            kernel_info['t_color'] = kernel_info_corr['t_color']
            kernel_info['t_kernel_build'] = prebuild_time  # Pre-build time (parallel)
            kernel_info['t_kernel_corr'] = kernel_info_corr['t_kernel_corr']  # Correlation time
            kernel_info['kernel_hits'] = kernel_info_corr['kernel_hits']
            kernel_info['kernel_misses'] = kernel_info_corr['kernel_misses']
            # [Phase 0 V0 fix — merge the gate-survival + kernel-eval counters
            # that were being computed inside compute_cost_matrix_gpu but
            # never copied back into the per-frame profile dict.]
            for _k in ('n_pairs_total', 'n_pairs_gated_dist',
                       'n_pairs_gated_combined', 'mean_cands_per_track',
                       'n_kernel_evals'):
                if _k in kernel_info_corr:
                    kernel_info[_k] = kernel_info_corr[_k]

            if early_locked_indices and C.size > 0:
                C[list(early_locked_indices), :] = 1e6

            # [In/outflow C/D] exited tracks take no detection; tracks that
            # coasted through an occluder only match near their predicted
            # re-emergence point (--occluder-gate radii).
            if _io_skip and C.size > 0:
                if _io_excl:
                    C[sorted(_io_excl), :] = 1e6
                if _io_occl:
                    _io_dxy = np.array([(c[0], c[1]) for c in all_candidates], dtype=np.float64)
                    _io_g = float(getattr(args, 'occluder_gate', 2.0))
                    for _i in _io_occl:
                        _far = np.hypot(_io_dxy[:, 0] - _io_px[_i],
                                        _io_dxy[:, 1] - _io_py[_i]) > _io_g * tracks[_i].r
                        C[_i, _far] = 1e6

            M, N = tracks_soa.count, len(all_candidates)

            # Phase 6: track Hungarian matrix sizes
            _prof_no_pkt_rows_total += M
            _prof_no_pkt_cols_total += N
            _hungarian_rows = M - len(early_locked_indices)
            _hungarian_cols = N
            _prof_hungarian_rows_total += _hungarian_rows
            _prof_hungarian_cols_total += _hungarian_cols
            _prof_hungarian_frames += 1

            # === [MAIN-9] Multi-stage assignment (see Flowcharts 9, 16) ===
            # Three-stage solver: (1) greedy_confident peels off unambiguous
            # matches, (2) solve_2x2_blocks handles confusable pairs with a
            # local 2x2 Hungarian, (3) solve_hungarian_stage3 runs the full
            # Hungarian on whatever remains. PKT-locked tracks are already
            # removed so stage 3's cost matrix is tiny on the primary video.
            # [Phase 4 Stage 4] Cascade Tier-1: iterative exclusion commits
            # before the assignment stages (cbtv.cascade). Flag off = no-op.
            _tier1_pairs = []
            if getattr(args, 'cascade_tier1', 0) == 1:
                from .cascade import tier1_exclusion_pass
                _t1_tr = [i for i in range(M) if i not in matched_tracks]
                _t1_dt = [j for j in range(N) if j not in matched_dets]
                _tier1_pairs, _ = tier1_exclusion_pass(
                    C, _t1_tr, _t1_dt,
                    margin=args.conf_margin, max_cost=args.conf_maxcost)
                matched_tracks.update(i for i, _ in _tier1_pairs)
                matched_dets.update(j for _, j in _tier1_pairs)

            # [Phase5-speed lever 1] sparse-gate selects byte-identical fast
            # variants of the two t_asn hotspots (greedy + 2x2). Default 0 ->
            # the shipped loop versions (byte-identical by being the same code).
            _sg = getattr(args, 'sparse_gate', 0) == 1
            _greedy = greedy_confident_sparse if _sg else greedy_confident
            _twox2 = solve_2x2_blocks_sparse if _sg else solve_2x2_blocks
            # [Phase5-speed] real-data parity probe (env CBTV_ASSIGN_PARITY=1):
            # snapshot the post-tier1 matched state so we can replay the whole
            # 3-stage with the OPPOSITE variant on this exact in-memory C and
            # assert identical pairs -- immune to the GPU cost-matrix run-to-run
            # noise that defeats a tracks.csv byte-compare. Zero cost when off.
            _parity_probe = bool(os.environ.get('CBTV_ASSIGN_PARITY') or os.environ.get('CPTV_ASSIGN_PARITY'))
            if _parity_probe:
                _mt0 = set(matched_tracks); _md0 = set(matched_dets)

            if args.use_3stage_matching:
                rem_tr_greedy = [i for i in range(M) if i not in matched_tracks]
                rem_dt_greedy = [j for j in range(N) if j not in matched_dets]

                if rem_tr_greedy and rem_dt_greedy:
                    p1, mt1, md1 = _greedy(C[np.ix_(rem_tr_greedy, rem_dt_greedy)], args.conf_margin, args.conf_maxcost)
                    # Translate indices back to original
                    p1 = [(rem_tr_greedy[i], rem_dt_greedy[j]) for (i,j) in p1]
                    matched_s1 = len(p1)
                    matched_tracks.update([i for i,_ in p1]); matched_dets.update([j for _,j in p1])
                else:
                    p1 = []
                    matched_s1 = 0

                rem_tr = [i for i in range(M) if i not in matched_tracks]
                rem_dt = [j for j in range(N) if j not in matched_dets]
                p2 = _twox2(C, rem_tr, rem_dt, max_cost=args.conf_maxcost*1.3)
                matched_s2 = len(p2)
                matched_tracks.update([i for i,_ in p2])
                matched_dets.update([j for _,j in p2])

                rem_tr = [i for i in range(M) if i not in matched_tracks]
                rem_dt = [j for j in range(N) if j not in matched_dets]
                p3 = solve_hungarian_stage3(C, rem_tr, rem_dt, max_cost=args.hungarian_max_cost)
                matched_s3 = len(p3)
                matched_tracks.update([i for i,_ in p3])
                matched_dets.update([j for _,j in p3])

                pairs = _tier1_pairs + p1 + p2 + p3

                if _linkdump_wr is not None:
                    for _i, _ in _tier1_pairs: _ld_stage[_i] = 'tier1'
                    for _i, _ in p1: _ld_stage[_i] = 'greedy'
                    for _i, _ in p2: _ld_stage[_i] = '2x2'
                    for _i, _ in p3: _ld_stage[_i] = 'hungarian'

                if _parity_probe:
                    # replay the 3 stages with the opposite (greedy/2x2) variant
                    # and the opposite Stage-3 prune on the identical C + state
                    _og = greedy_confident if _sg else greedy_confident_sparse
                    _ox = solve_2x2_blocks if _sg else solve_2x2_blocks_sparse
                    _mt, _md = set(_mt0), set(_md0)
                    _rtg = [i for i in range(M) if i not in _mt]
                    _rdg = [j for j in range(N) if j not in _md]
                    if _rtg and _rdg:
                        _q1, _, _ = _og(C[np.ix_(_rtg, _rdg)], args.conf_margin, args.conf_maxcost)
                        _q1 = [(_rtg[i], _rdg[j]) for (i, j) in _q1]
                    else:
                        _q1 = []
                    _mt.update(i for i, _ in _q1); _md.update(j for _, j in _q1)
                    _rt = [i for i in range(M) if i not in _mt]
                    _rd = [j for j in range(N) if j not in _md]
                    _q2 = _ox(C, _rt, _rd, max_cost=args.conf_maxcost * 1.3)
                    _mt.update(i for i, _ in _q2); _md.update(j for _, j in _q2)
                    _rt = [i for i in range(M) if i not in _mt]
                    _rd = [j for j in range(N) if j not in _md]
                    _q3 = solve_hungarian_stage3(C, _rt, _rd, max_cost=args.hungarian_max_cost)
                    _a = {(int(i), int(j)) for i, j in (_tier1_pairs + p1 + p2 + p3)}
                    _b = {(int(i), int(j)) for i, j in (_tier1_pairs + _q1 + _q2 + _q3)}
                    if _a != _b:
                        import sys as _sys
                        # classify: equal TOTAL assignment cost -> benign tie
                        # (choice among equally-optimal candidates); else a real
                        # cost-changing divergence (a bug).
                        _ca = sum(float(C[i, j]) for i, j in _a)
                        _cb = sum(float(C[i, j]) for i, j in _b)
                        _benign = abs(_ca - _cb) < 1e-6
                        kind = 'BENIGN-TIE' if _benign else 'REAL-MISMATCH'
                        globals()['_ASSIGN_PARITY_BENIGN'] = globals().get('_ASSIGN_PARITY_BENIGN', 0) + int(_benign)
                        globals()['_ASSIGN_PARITY_REAL'] = globals().get('_ASSIGN_PARITY_REAL', 0) + int(not _benign)
                        print(f"[ASSIGN_PARITY] FRAME {processed}: {kind} "
                              f"sym_diff={sorted(_a ^ _b)[:8]} costΔ={_ca - _cb:+.6g} "
                              f"(|a|={len(_a)} |b|={len(_b)})", file=_sys.stderr, flush=True)
                    else:
                        globals()['_ASSIGN_PARITY_OK'] = globals().get('_ASSIGN_PARITY_OK', 0) + 1
            else:
                rem_tr_greedy = [i for i in range(M) if i not in matched_tracks]
                rem_dt_greedy = [j for j in range(N) if j not in matched_dets]

                if rem_tr_greedy and rem_dt_greedy:
                    p1, mt, md = _greedy(C[np.ix_(rem_tr_greedy, rem_dt_greedy)], args.conf_margin, args.conf_maxcost)
                    # Translate indices back to original
                    p1 = [(rem_tr_greedy[i], rem_dt_greedy[j]) for (i,j) in p1]
                    matched_s1 = len(p1)
                    matched_tracks.update([i for i,_ in p1]); matched_dets.update([j for _,j in p1])
                else:
                    p1 = []
                    matched_s1 = 0

                rem_tr = [i for i in range(M) if i not in matched_tracks]
                rem_dt = [j for j in range(N) if j not in matched_dets]
                p2 = _twox2(C, rem_tr, rem_dt, max_cost=args.conf_maxcost*1.3)
                matched_s2 = len(p2)
                matched_tracks.update([i for i,_ in p2])
                matched_dets.update([j for _,j in p2])

                pairs = _tier1_pairs + p1 + p2

                if _linkdump_wr is not None:
                    for _i, _ in _tier1_pairs: _ld_stage[_i] = 'tier1'
                    for _i, _ in p1: _ld_stage[_i] = 'greedy'
                    for _i, _ in p2: _ld_stage[_i] = '2x2'

            for (i,j) in pairs:
                t=tracks[i]
                x,y,r,s = all_candidates[j]
                t.px, t.py, getattr(t, 'pr', None)
                t.px, t.py, t.pr = t.x, t.y, t.r
                t.x,t.y,t.r = x,y, 0.85*t.r+0.15*r
                old_c = np.array(t.color_bgr, dtype=np.float64)
                new_c = np.array(colors_all[j], dtype=np.float64)
                # [Phase 4C / Phase 0] Color EMA: beta new + (1-beta) old.
                # beta is now CLI-controlled via --colour-ema (default 0.3, matches
                # the locked baseline). beta=1.0 reproduces the historic "wholesale
                # overwrite" bug that drove L01 jitter oscillation; smaller beta
                # smooths harder. Bounded to [0,1] for safety.
                _beta = float(getattr(args, 'colour_ema', 0.3))
                if _beta < 0.0: _beta = 0.0
                elif _beta > 1.0: _beta = 1.0
                t.color_bgr = tuple((_beta * new_c + (1.0 - _beta) * old_c).tolist())
                t.missed=0
                t.last_det_pos = (x, y)
                t.last_det_r = r

        t_asn=(time.perf_counter()-t_asn0)*1000.0

        # ========================================================================
        # SECOND PASS — re-match with relaxed thresholds (see Flowchart 9: Assignment Pipeline)
        # ========================================================================
        t_sp2=0.0; sp2_matches=0
        if args.second_pass and tracks:
            t_sp20=time.perf_counter()

            left_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
            if _io_skip:
                left_tracks = [i for i in left_tracks if i not in _io_skip]
            unmatched_indices = [j for j in range(len(all_candidates)) if j not in matched_dets]
            cand2 = [all_candidates[j] for j in unmatched_indices]

            if left_tracks and cand2:
                # GPU-ACCELERATED: Batch color extraction for second pass
                colors2 = batch_compute_colors_gpu(frame, cand2, use_gpu=args.use_gpu)

                # Create subset of tracks for second pass
                left_track_objs = [tracks[i] for i in left_tracks]
                tracks_soa2 = TrackArrays(left_track_objs)

                # Create temporary args with relaxed thresholds for second pass
                class SecondPassArgs:
                    """Relaxed argument container for second-pass matching."""
                    def __init__(self, args_orig, md_factor, mc_factor):
                        """Initialize with relaxed distance and color thresholds."""
                        self.max_dist = args_orig.max_dist * md_factor
                        self.max_color = args_orig.max_color * mc_factor
                        self.w_dist = args_orig.w_dist
                        self.w_color = args_orig.w_color
                        self.gamma = args_orig.gamma
                        self.use_neighborhood_kernels = args_orig.use_neighborhood_kernels
                        self.neighbor_search_factor = args_orig.neighbor_search_factor
                        self.min_neighbors = args_orig.min_neighbors
                        self.sigma_factor_central = args_orig.sigma_factor_central
                        self.sigma_factor_neighbour = args_orig.sigma_factor_neighbour
                        self.kernel_margin = args_orig.kernel_margin
                        self.search_factor = args_orig.search_factor
                        self.adaptive_gamma = args_orig.adaptive_gamma
                        self.color_saturation_weight = args_orig.color_saturation_weight
                        self.color_nonlinear_k = args_orig.color_nonlinear_k
                        self.glare_threshold = args_orig.glare_threshold
                        self.persistence_bonus = args_orig.persistence_bonus
                        self.color_source = getattr(args_orig, 'color_source', 'frame')

                args2 = SecondPassArgs(args, args.second_pass_factor, args.second_color_factor)

                # GPU-ACCELERATED: Vectorized cost matrix for second pass
                C2, _ = compute_cost_matrix_gpu(
                    tracks_soa2, cand2, colors2, frame, frame_lab,
                    previous_frame, previous_detections, kernel_cache, args2,
                    use_gpu=args.use_gpu
                )

                M2, N2 = len(left_tracks), len(cand2)

                if hungarian is not None:
                    pairs2 = solve_hungarian_stage3(C2, range(M2), range(N2), max_cost=args.hungarian_max_cost)
                else:
                    pairs2 = []
                    used_c = set()
                    for ii in range(M2):
                        jj = int(np.argmin(C2[ii]))
                        if C2[ii, jj] < args.hungarian_max_cost and jj not in used_c:
                            pairs2.append((ii, jj))
                            used_c.add(jj)

                for (ii, jj) in pairs2:
                    i = left_tracks[ii]
                    orig_j = unmatched_indices[jj]
                    x, y, r, s = cand2[jj]
                    t = tracks[i]
                    t.px, t.py, t.pr = t.x, t.y, t.r
                    t.x, t.y, t.r = x, y, 0.85 * t.r + 0.15 * r
                    old_c = np.array(t.color_bgr, dtype=np.float64)
                    new_c = np.array(colors2[jj], dtype=np.float64)
                    # [Phase 4C / Phase 0] Same --colour-ema beta as the first pass.
                    _beta = float(getattr(args, 'colour_ema', 0.3))
                    if _beta < 0.0: _beta = 0.0
                    elif _beta > 1.0: _beta = 1.0
                    t.color_bgr = tuple((_beta * new_c + (1.0 - _beta) * old_c).tolist())
                    tracks[i].missed = 0
                    t.last_det_pos = (x, y)
                    t.last_det_r = r
                    matched_tracks.add(i)
                    matched_dets.add(orig_j)
                    sp2_matches += 1
                    if _linkdump_wr is not None:
                        _ld_stage[i] = 'sp2'

            t_sp2 = (time.perf_counter() - t_sp20) * 1000.0

        # ========================================================================
        # KERNEL RECOVERY — batched re-detection (see Flowchart 10: Track Lifecycle)
        # ========================================================================
        t_krec=0.0; kernel_recoveries=0
        virtual_detections = []

        # [Phase5-speed lever 2] kernel-recovery gate. With --kernel-recovery-gate 1,
        # skip the 150-247 ms NCC recovery fallback when the cost matrix's kernel
        # term is off (w_kernel==0). Default 0 keeps the gate condition identical
        # to the shipped path (recovery always runs) -> byte-identical.
        _krec_gate_on = getattr(args, 'kernel_recovery_gate', 0) == 1
        if (args.kernel_recovery and tracks and previous_frame is not None
                and previous_detections is not None
                and (not _krec_gate_on or float(getattr(args, 'w_kernel', 0.0)) > 0.0)):
            t_krec0=time.perf_counter()

            left_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
            if _io_skip:
                left_tracks = [i for i in left_tracks if i not in _io_skip]

            # BATCHED kernel recovery (converts LAB once, builds kernels in batch)
            # Phase 3l fix: pass FRESH current-frame detections so recovery
            # can snap NCC peaks to nearest fresh detection coords (not
            # write the NCC peak directly).
            _all_curr_dets = list(kept) + list(soft)
            recoveries = batch_kernel_recovery(
                left_tracks, tracks, previous_detections, previous_frame,
                frame, args, current_detections=_all_curr_dets
            )

            # Apply recoveries
            for track_idx, det_x, det_y, det_r, peak_val in recoveries:
                t = tracks[track_idx]
                t.px, t.py, t.pr = t.x, t.y, t.r
                t.x, t.y = det_x, det_y
                # Keep existing t.r (carries over)
                # Keep existing t.color_bgr (carries over)
                t.missed = 0
                t.last_det_pos = (det_x, det_y)
                t.last_det_r = det_r
                matched_tracks.add(track_idx)

                # Virtual detection for previous_detections
                virtual_det = (det_x, det_y, det_r, peak_val)
                virtual_detections.append(virtual_det)
                kernel_recoveries += 1
                if _linkdump_wr is not None:
                    _ld_stage[track_idx] = 'krec'

            t_krec = (time.perf_counter() - t_krec0) * 1000.0

        # [Phase 6g Stage 1] linkdump write: one row per track that existed at
        # prediction time. Runs BEFORE [MAIN-10] so list indices still align
        # with the snapshot (spawn/prune has not mutated `tracks` yet).
        if _linkdump_wr is not None and _ld_pre is not None:
            for _li, (_tid, _ox, _oy, _lpx, _lpy, _lsrc, _lalpha, _lvel,
                      _lmiss, _lng, _lns) in enumerate(_ld_pre):
                _lt = tracks[_li]
                _lm = 1 if _li in matched_tracks else 0
                if _lm:
                    _lstage = _ld_stage.get(_li, '?')
                    _ldx, _ldy = float(_lt.x), float(_lt.y)
                    _lgd = float(np.hypot(_ldx - _lpx, _ldy - _lpy))
                    _linkdump_wr.writerow(
                        [processed, _tid, f"{_ox:.2f}", f"{_oy:.2f}",
                         f"{_lpx:.2f}", f"{_lpy:.2f}", _lsrc, f"{_lalpha:.3f}",
                         _lvel, _lmiss, 1, _lstage,
                         f"{_ldx:.2f}", f"{_ldy:.2f}", f"{_lgd:.2f}",
                         _lng, _lns])
                else:
                    _linkdump_wr.writerow(
                        [processed, _tid, f"{_ox:.2f}", f"{_oy:.2f}",
                         f"{_lpx:.2f}", f"{_lpy:.2f}", _lsrc, f"{_lalpha:.3f}",
                         _lvel, _lmiss, 0, 'miss', "", "", "",
                         _lng, _lns])

        # ========================================================================
        # === [MAIN-10] Track lifecycle: spawn + age + prune (see Flowchart 10) ===
        # Unmatched detections spawn new tracks (capped when PKT is active);
        # unmatched tracks have their missed counter incremented, boundary
        # tracks get velocity-based extrapolation and extended tolerance, and
        # tracks exceeding miss_tolerance are pruned. This is the only path
        # that mutates the `tracks` list length between frames.
        # SPAWN PHASE (see Flowchart 10: Track Lifecycle — spawn cap + boundary awareness)
        # ========================================================================
        # Spawn new tracks for truly unmatched detections
        # [Phase 4D] Boundary-aware tracking: particles within BOUNDARY_MARGIN of frame edge
        # get extra miss tolerance (BOUNDARY_MISS_BONUS) and velocity-based extrapolation
        # because they may temporarily exit and re-enter the field of view.
        # [Decision Rationale] 20px ≈ 2× particle diameter (see Flowchart 10 — Track Lifecycle): particles at the edge have
        # partial visibility and are more likely to be missed by NCC correlation.
        # [In/outflow F] --boundary-margin (default 20 px, the old constant)
        BOUNDARY_MARGIN = args.boundary_margin  # pixels from frame edge (used by spawn + lifecycle)
        # [SPAWN-1] Spawn timer + PKT coverage calculation
        spawned = 0
        # [v6 Change] PKT coverage gating and spawn cap are v6-new — original had no spawn limit.
        # When PKT locked most tracks, cap spawns per frame to prevent FP accumulation
        _pkt_coverage = len(early_locked_indices) / max(1, len(tracks)) if tracks else 0.0
        # [SPAWN-2] Spawn cap decision
        _pkt_spawn_cap = 5 if _pkt_coverage > 0.5 else 999999  # Cap at 5 when PKT active
        # Find all detections that were not matched in any stage
        unmatched_indices_all = [j for j in range(len(all_candidates)) if j not in matched_dets]
        _io_no_live = len(tracks) == 0   # [In/outflow] birth_reason 'restart'
        # [SPAWN-3] Unmatched detection loop
        for j in unmatched_indices_all:
            x,y,r,s = all_candidates[j]
            # Cap spawns when PKT is active to prevent FP accumulation
            if spawned >= _pkt_spawn_cap:
                break
            # [SPAWN-4] High-confidence / first-frame exception gate
            # ONLY spawn new tracks from HIGH confidence (hard) detections
            # Exception: On the very first frame, use all surviving detections to ensure high initial recall
            # Also allow spawning from soft candidates if there are no tracks yet (e.g., first frame after all tracks died)
            if j < kept_len or processed == 0 or len(tracks) == 0:
                # [SPAWN-5] Track construction + ID assignment
                track_idx = len(tracks)  # Index of the new track
                tracks.append(Track(id=next_id, x=x, y=y, r=r, color_bgr=colors_all[j], missed=0, px=None, py=None, pr=None))
                tracks[-1].birth_frame = processed
                if _io_region is not None:
                    tracks[-1].birth_reason = _io_birth_reason(
                        processed, x, y, r, _io_region, BOUNDARY_MARGIN,
                        float(getattr(args, 'occluder_gate', 2.0)), _io_no_live)
                    # [In/outflow E] tentative until k_need consecutive matches
                    tracks[-1].k_need = _io_Ke if tracks[-1].birth_reason == 'entry_band' else _io_K
                    tracks[-1].tentative = tracks[-1].k_need > 1
                    tracks[-1].hits = 1
                # [SPAWN-6] matched_tracks bookkeeping
                matched_tracks.add(track_idx)  # Mark spawned track as matched (prevent aging on spawn frame)
                next_id += 1
                spawned += 1

        # [--color-source track] store each matched / spawned track's colour at
        # its new (detected) position, EMA'd with --colour-ema. The cost
        # matrix then compares detections with this instead of re-sampling
        # the next frame at the track's old position. Off by default.
        if getattr(args, 'color_source', 'frame') in ('track', 'min') and matched_tracks:
            from .color import sample_weighted_colors
            _ct_idx = [i for i in sorted(matched_tracks) if i < len(tracks)]
            _ct_lab, _ct_S = sample_weighted_colors(
                frame, [(tracks[i].x, tracks[i].y, tracks[i].r, 0.0) for i in _ct_idx],
                glare_threshold=args.glare_threshold)
            _ct_b = min(1.0, max(0.0, float(getattr(args, 'colour_ema', 0.3))))
            for _k, _i in enumerate(_ct_idx):
                _t = tracks[_i]
                if _t.lab_w is None:
                    _t.lab_w, _t.S_w = tuple(_ct_lab[_k]), float(_ct_S[_k])
                else:
                    _t.lab_w = tuple(_ct_b * _ct_lab[_k] + (1.0 - _ct_b) * np.asarray(_t.lab_w))
                    _t.S_w = _ct_b * float(_ct_S[_k]) + (1.0 - _ct_b) * _t.S_w

        # ========================================================================
        # STORE PREVIOUS FRAME DATA
        # Include real detections + virtual (kernel-recovered) detections
        # ========================================================================
        previous_detections = [(d[0], d[1], d[2], d[3]) for d in all_candidates] if tracks else []
        if args.kernel_recovery:
            previous_detections.extend(virtual_detections)
        previous_frame = frame.copy()

        # [PRUNE-1] Miss counter increment + velocity decay
        # AGE & PRUNE PHASE (see Flowchart 10: Track Lifecycle — boundary extrapolation + miss tolerance)
        # [Phase 4D] Age & prune (with boundary-aware extrapolation)
        # [Decision Rationale] BOUNDARY_MISS_BONUS=6 (see Flowchart 10 — Track Lifecycle): particles near the frame edge may disappear
        # for several frames as they exit the field of view. 6 extra frames (~0.2s at 30fps)
        # prevents premature track death and re-spawning (which would create a new track ID).
        # Combined with miss_tolerance=8, boundary tracks survive up to 14 frames unmatched.
        # [v6 Change] Boundary-aware extrapolation and BOUNDARY_MISS_BONUS are v6-new —
        # original had flat miss_tolerance with no edge-aware logic.
        if _io_on:
            # [In/outflow B-E] coast prediction, exit kill, occluder coast,
            # tentative births (cbtv.inoutflow.io_lifecycle).
            from .inoutflow import io_lifecycle
            tracks = io_lifecycle(tracks, matched_tracks, _io_excl, processed, args,
                                  _io_region, prior_positions=predicted_positions,
                                  prior_ids=affine_by_id)
        else:
            BOUNDARY_MISS_BONUS = args.boundary_miss_bonus  # extra frames of tolerance for boundary tracks (default 6)
            for i, t in enumerate(tracks):
                if i not in matched_tracks:
                    t.missed += 1
                    t.last_det_pos = None
                    t.last_det_r = None
                    t.coast_state = 'coast_frozen'  # [In/outflow G] row tag
                    # [PRUNE-2] Boundary-aware extrapolation
                    # Boundary-aware extrapolation: if track is near frame edge and has
                    # velocity, extrapolate position so it still appears in output CSV.
                    if t.px is not None and t.py is not None:
                        vx = t.x - t.px
                        vy = t.y - t.py
                        pred_x = t.x + vx
                        pred_y = t.y + vy
                        near_edge = (pred_x < BOUNDARY_MARGIN or pred_x > W - BOUNDARY_MARGIN or
                                     pred_y < BOUNDARY_MARGIN or pred_y > H - BOUNDARY_MARGIN)
                        if near_edge and t.missed <= 3:
                            t.px, t.py = t.x, t.y
                            t.x = float(np.clip(pred_x, 1, W - 2))
                            t.y = float(np.clip(pred_y, 1, H - 2))
                            t.coast_state = 'coast_pred'
            # [PRUNE-3] BOUNDARY_MISS_BONUS tolerance widening
            # Extended tolerance for boundary tracks; normal tolerance otherwise
            surviving = []
            for t in tracks:
                near_edge = (t.x < BOUNDARY_MARGIN or t.x > W - BOUNDARY_MARGIN or
                             t.y < BOUNDARY_MARGIN or t.y > H - BOUNDARY_MARGIN)
                tolerance = args.miss_tolerance + BOUNDARY_MISS_BONUS if near_edge else args.miss_tolerance
                # [PRUNE-4] Track deletion when miss > tolerance
                if t.missed <= tolerance:
                    surviving.append(t)
                else:
                    t.death_reason = 'miss_timeout'
            tracks = surviving

        # Debug first frame
        if processed == 0 and args.dump_first_frame_corr:
            os.makedirs(args.dump_dir, exist_ok=True)
            vis = frame.copy()
            for (x, y, r, s) in kept:
                cv2.circle(vis, (int(round(x)), int(round(y))), int(max(2, round(r))), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.dump_dir, "F0_overlay_kept.png"), vis)
            Path(os.path.join(args.dump_dir, "F0_dbg.json")).write_text(json.dumps(dbg, indent=2))

        # === [MAIN-11] CSV + annotated video frame output (see Flowchart 18 row "Output") ===
        # Per-track rows are appended to the output CSV and the annotated
        # frame is written to the VideoWriter. This is the only place the
        # tracker emits user-visible per-frame data; the evaluator consumes
        # the CSV downstream via the positional-args path in Flowchart 18.
        # Draw & write
        vis = frame.copy()
        for (x, y, r, s) in kept:
            cv2.circle(vis, (int(round(x)), int(round(y))), int(max(2, round(r))), (0, 255, 0), 2)
        for idx, t in enumerate(tracks):
            color = (255, 100, 0) if t.missed == 0 else (140, 140, 140)
            cv2.circle(vis, (int(round(t.x)), int(round(t.y))), int(max(2, round(t.r))), color, 2)
            cv2.putText(vis, str(t.id), (int(t.x) + 3, int(t.y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if _io_rows is None:
                # [csv-trust] matched rows report the detection's own radius
                _r_out = (t.last_det_r if (_csv_trust and t.missed == 0 and t.last_det_r is not None)
                          else t.r)
                csv_wr.writerow([processed, t.id, f"{t.x:.3f}", f"{t.y:.3f}", f"{_r_out:.3f}", t.missed])
        if _io_rows is not None:
            _io_rows.frame(processed, tracks)
        writer.write(vis)

        total = (time.perf_counter() - t0) * 1000.0
        if args.verbose:
            print(f"[F{processed}] thr={thr:.3f} raw={dbg.get('raw',0)} +bayes={dbg.get('bayes_added',0)} dedup={len(deduped)} refined={len(refined)} kept={len(kept)} soft={len(soft)} tracks={len(tracks)} matched={len(matched_tracks)} (s1={matched_s1} s2={matched_s2} s3={matched_s3}) sp2={sp2_matches} krec={kernel_recoveries} spawned={spawned} prev_dets={len(previous_detections)}")
            print(f"  [TIME] Timing: t_det={t_det:.1f} t_filt={t_filt:.1f} t_cap={t_cap:.1f} t_color={t_color_extract:.1f} t_asn={t_asn:.1f} (dist={kernel_info['t_dist']:.1f} color={kernel_info['t_color']:.1f} kbuild={kernel_info['t_kernel_build']:.1f} kcorr={kernel_info['t_kernel_corr']:.1f}) t_sp2={t_sp2:.1f} t_krec={t_krec:.1f} total={total:.1f}ms")
            if kernel_info['kernel_hits'] > 0 or kernel_info['kernel_misses'] > 0:
                print(f"  [CACHE] Kernel Cache: hits={kernel_info['kernel_hits']} misses={kernel_info['kernel_misses']} hit_rate={100.0*kernel_info['kernel_hits']/(kernel_info['kernel_hits']+kernel_info['kernel_misses']+1e-9):.1f}%")

        # Phase 6: per-frame profiling summary (always printed when verbose)
        # [Phase 6] Per-stage timing instrumentation for profiling
        if getattr(args, 'verbose', False):
            _costmat_ms = kernel_info['t_dist'] + kernel_info['t_color'] + kernel_info['t_kernel_build'] + kernel_info['t_kernel_corr']
            _update_ms = total - t_det - t_pkt_ms - _costmat_ms - t_sp2 - t_krec
            print(f"[PROFILE] Frame {processed}: detect={t_det:.0f}ms pkt={t_pkt_ms:.0f}ms(locked={_pkt_n_locked}/{len(tracks)}) "
                  f"costmat={_costmat_ms:.0f}ms hungarian={t_asn:.0f}ms(size={_hungarian_rows}x{_hungarian_cols}) "
                  f"update={_update_ms:.0f}ms TOTAL={total:.0f}ms")

        if prof_wr:
            prof_wr.writerow([processed, dbg.get('raw', 0), dbg.get('bayes_added', 0), len(deduped),
                             len(kept), len(soft), len(tracks), matched_s1, matched_s2, matched_s3,
                             spawned, sp2_matches, kernel_recoveries,
                             f"{thr:.3f}",
                             f"{t_det:.1f}", f"{t_filt:.1f}", f"{t_cap:.1f}",
                             f"{t_color_extract:.1f}", f"{t_pkt_ms:.1f}",
                             f"{kernel_info['t_dist']:.1f}", f"{kernel_info['t_color']:.1f}",
                             f"{kernel_info['t_kernel_build']:.1f}", f"{kernel_info['t_kernel_corr']:.1f}",
                             kernel_info['kernel_hits'], kernel_info['kernel_misses'],
                             # [Phase 0 V0] gate-survival + kernel-eval + PKT counters
                             kernel_info.get('n_pairs_total', 0),
                             kernel_info.get('n_pairs_gated_dist', 0),
                             kernel_info.get('n_pairs_gated_combined', 0),
                             f"{kernel_info.get('mean_cands_per_track', 0.0):.3f}",
                             kernel_info.get('n_kernel_evals', 0),
                             _pkt_n_locked, _pkt_soft_locked, _pkt_low_ncc_skipped,
                             _pkt_backwards_this_frame,
                             _field_alpha_pos,
                             f"{(_field_alpha_sum / max(1, _field_alpha_pos)):.4f}",
                             _n_easy_committed,
                             f"{(_field_n_anchors_sum / max(1, _field_alpha_pos)):.2f}",
                             # [Phase 3 Stage C] constellation diagnostics
                             _const_stats["proposed"], _const_stats["accepted"],
                             _const_stats["members_committed"], _const_stats["outliers"],
                             _const_stats["field_refreshed"], f"{t_const:.1f}",
                             f"{t_asn:.1f}", f"{t_sp2:.1f}", f"{t_krec:.1f}", f"{total:.1f}"])

        processed += 1
        if args.max_frames > 0 and processed >= args.max_frames: break
        if pbar:
            try:
                pbar.update(1)
            except Exception:
                pass

    # === [MAIN-12] End-of-run cleanup + metrics summary (see Flowchart 18) ===
    # Release video handles, flush diagnostics, and print the Phase 6 PKT
    # profiling summary (full-lock / soft-lock / skip percentages and the
    # average Hungarian matrix size). The evaluator is invoked externally
    # after main() returns; see Flowchart 18 for what it computes.
    cap.release()
    writer.release()
    if _io_rows is not None:
        _io_rows.finish(tracks)
    csv_fp.close()
    if _linkdump_fp is not None:
        _linkdump_fp.close()
    if prof_wr:
        pf.close()
    if _cluster_log_fp is not None:
        _cluster_log_fp.close()
    if diag_fp is not None:
        diag_fp.close()
        print(f"[DIAGNOSTIC] Saved diagnostic_log.csv ({processed} rows)")
    if pbar:
        try:
            pbar.close()
        except Exception:
            pass
    print('Saved:', args.output_video, 'and', args.output_csv)

    if os.environ.get('CBTV_ASSIGN_PARITY') or os.environ.get('CPTV_ASSIGN_PARITY'):
        _ok = globals().get('_ASSIGN_PARITY_OK', 0)
        _bn = globals().get('_ASSIGN_PARITY_BENIGN', 0)
        _rl = globals().get('_ASSIGN_PARITY_REAL', 0)
        print(f"[ASSIGN_PARITY] frames: {_ok} identical, {_bn} benign-tie "
              f"(equal-cost), {_rl} REAL-mismatch (cost-changing). "
              f"{'PASS (no cost-changing divergence)' if _rl == 0 else '*** FAIL: REAL MISMATCH ***'}")

    # Phase 6: PKT profiling summary
    if _prof_pkt_active_total > 0:
        _pf = _prof_pkt_full_lock_total
        _ps = _prof_pkt_soft_lock_total
        _pk = _prof_pkt_skip_total
        _pa = _prof_pkt_active_total
        _pct_full = 100.0 * _pf / _pa
        _pct_soft = 100.0 * _ps / _pa
        _pct_skip = 100.0 * _pk / _pa
        _avg_h_r = _prof_hungarian_rows_total / max(1, _prof_hungarian_frames)
        _avg_h_c = _prof_hungarian_cols_total / max(1, _prof_hungarian_frames)
        _avg_nopkt_r = _prof_no_pkt_rows_total / max(1, _prof_hungarian_frames)
        _avg_nopkt_c = _prof_no_pkt_cols_total / max(1, _prof_hungarian_frames)
        print(f"\n[PKT SUMMARY] Across {processed} frames: avg full_lock={_pct_full:.1f}% soft_lock={_pct_soft:.1f}% skip={_pct_skip:.1f}% | "
              f"Avg Hungarian size: {_avg_h_r:.0f}x{_avg_h_c:.0f} (without PKT would be {_avg_nopkt_r:.0f}x{_avg_nopkt_c:.0f})")
        # [Phase 0] Backwards-lock magnitude. Non-zero count means PKT moved at least one
        # track against its velocity vector — the signature of the cached-template bug.
        # Under the default (--pkt-rebuild-template=1) this should be small.
        print(f"[PKT DIAG] Backwards-lock events: {_prof_pkt_backwards_lock_total} "
              f"({100.0 * _prof_pkt_backwards_lock_total / max(1, _prof_pkt_full_lock_total):.2f}% of full locks)")
    else:
        print(f"\n[PKT SUMMARY] PKT was not active during this run ({processed} frames)")


if __name__ == "__main__":
    main()
