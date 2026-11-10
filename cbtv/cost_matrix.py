"""Cost-matrix computation (distance + colour + kernel-NCC)."""
import os
import time

import numpy as np

from .color import batch_smart_color_distance_gpu, vectorized_distance_matrix_gpu
from .gpu import cp


def adaptive_weight_terms(w_dist_base, w_color_base, gamma_base, S_conf_batch,
                          neigh_counts_batch=None, min_neighbors_legacy=2, xp=np):
    """Vectorized port of ``compute_adaptive_weights_normalized``
    (hybrid_tracker.py:4360), extracted as a pure function (Phase 3 Stage A)
    so it is unit-testable against the scalar original on CPU.

    Parameters
    ----------
    w_dist_base, w_color_base, gamma_base : float
        Base weights (args.w_dist / args.w_color / args.gamma).
    S_conf_batch : (B,) array
        Per-pair colour saturation confidence.
    neigh_counts_batch : (B,) array or None
        Per-pair neighbour count of the pair's *track* (the count that
        built the track's kernel). When given, neigh_factor follows the
        scalar original's per-track thresholds (>=5 -> 1.2, >=3 -> 1.0,
        else 0.9). ``None`` reproduces the legacy (pre-Stage-A) vectorized
        behaviour that used the constant ``args.min_neighbors`` — the
        port wart this function fixes.
    xp : numpy or cupy module.

    Returns
    -------
    (w_dist_final, w_color_effective, gamma_final) — scalars and/or (B,)
    arrays broadcastable against S_conf_batch.

    Deviations from the scalar original (shipped GCS-vectorization
    behaviour, intentionally preserved):
      * kernel_conf is pinned at 1.0 (point-wise NCC has no confidence
        map) -> conf_factor = 1.5.
      * The original's [W-3] step (redistributing the gamma delta into
        w_dist/w_color) is NOT applied: with gamma_base = 8.0 the delta
        (~ +2.8 vs w_dist + w_color = 1.0) would drive w_dist negative.
        Only the [W-2] neigh_factor selection and the [W-4]
        low-saturation colour redistribution are live.
    """
    # [W-2] neigh_factor — per-track when counts are given (wart fix).
    if neigh_counts_batch is not None:
        nc = xp.asarray(neigh_counts_batch)
        neigh_factor = xp.where(nc >= 5, 1.2, xp.where(nc >= 3, 1.0, 0.9))
    else:
        if min_neighbors_legacy >= 5: neigh_factor = 1.2
        elif min_neighbors_legacy >= 3: neigh_factor = 1.0
        else: neigh_factor = 0.9

    # point-wise NCC simplifies things, kernel_conf = 1.0 -> conf_factor = 1.5
    conf_factor = 1.5
    gamma_adjusted = gamma_base * conf_factor * neigh_factor

    w_dist_after_gamma = w_dist_base
    w_color_after_gamma = w_color_base

    # [W-4] low-saturation colour-weight redistribution (identical algebra
    # to the scalar original given w_dist_after_gamma / gamma_adjusted).
    w_color_effective = w_color_after_gamma * S_conf_batch
    color_weight_lost = w_color_after_gamma * (1.0 - S_conf_batch)

    non_color_base = w_dist_after_gamma + gamma_adjusted
    if neigh_counts_batch is None:
        # Legacy scalar branch — kept verbatim for byte-identical parity.
        if non_color_base > 0:
            w_dist_final = w_dist_after_gamma + color_weight_lost * (w_dist_after_gamma / non_color_base)
            gamma_final = gamma_adjusted + color_weight_lost * (gamma_adjusted / non_color_base)
        else:
            w_dist_final = w_dist_after_gamma + color_weight_lost
            gamma_final = gamma_adjusted
    else:
        safe = non_color_base > 0
        denom = xp.where(safe, non_color_base, 1.0)
        w_dist_final = xp.where(
            safe,
            w_dist_after_gamma + color_weight_lost * (w_dist_after_gamma / denom),
            w_dist_after_gamma + color_weight_lost)
        gamma_final = xp.where(
            safe,
            gamma_adjusted + color_weight_lost * (gamma_adjusted / denom),
            gamma_adjusted)
    return w_dist_final, w_color_effective, gamma_final


def _maybe_dump_C(dump_frame, C, tracks_soa, N, valid_pairs, detections=None):
    """Parity debugging: with env CBTV_DUMP_C=<dir>, write the final per-frame
    cost matrix to <dir>/C_<frame:05d>.npz (C as float64, track ids = row
    order, detection indices = column order, valid_pairs mask). No-op when the
    variable is unset or the caller passed no frame index."""
    if dump_frame is None:
        return
    out_dir = os.environ.get("CBTV_DUMP_C") or os.environ.get("CPTV_DUMP_C")  # CPTV_*: former name
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    M = tracks_soa.count
    if valid_pairs is None:
        valid_pairs = np.zeros((M, N), dtype=bool)
    np.savez(os.path.join(out_dir, f"C_{int(dump_frame):05d}.npz"),
             C=np.array(C, dtype=np.float64),
             track_ids=np.asarray(tracks_soa.ids, dtype=np.int64)[:M],
             det_idx=np.arange(N, dtype=np.int64),
             valid_pairs=np.asarray(valid_pairs, dtype=bool),
             # column j's detection (x, y, r): lets an audit map C to detections.
             det_xyr=np.array([[d[0], d[1], d[2]] for d in (detections or [])],
                              dtype=np.float64).reshape(-1, 3))


def compute_cost_matrix_gpu(tracks_soa, detections, det_colors, frame_bgr, frame_lab,
                           prev_frame, prev_detections, kernel_cache, args, use_gpu=True, frame_lab_gpu=None,
                           predicted_positions=None, neighbor_counts=None, graph_struct_data=None,
                           dump_frame=None):
    # [v6 Change] 5-step cost matrix pipeline (see Flowchart 8 — Cost Matrix):
    #   Original: 3 components (distance + color + kernel NCC) with fixed weights.
    #   Current: 5 components adding persistence bonus (Phase 5-D) and velocity
    #   prediction (Phase 5-F). Persistence bonus uses cKDTree spatial match at
    #   1.5× radius to identify re-assignments to the same detection, applying a
    #   0.85 cost multiplier (15% reduction). Velocity prediction shifts expected
    #   track positions by EMA velocity for tracks moving > 2.0 px/frame.
    #   Gather-Correlate-Scatter GPU path (Phase 4) batches kernel NCC across
    #   all valid pairs in a single GPU pass instead of per-pair CPU loops.
    """
    Per-frame track-to-detection cost matrix (GPU, with a numpy CPU path).

    This replaces the nested O(M×N) loop with vectorized GPU operations:
    - Vectorized distance matrix
    - Batch color distance computation
    - Cached kernel lookups
    - Hard gates applied early to avoid expensive operations

    Args:
        tracks_soa: TrackArrays object with track data
        detections: List of (x, y, r, score) tuples
        det_colors: List of (b, g, r) tuples (precomputed)
        frame_bgr: Current BGR frame
        frame_lab: Current LAB frame (for kernel correlation)
        prev_frame: Previous BGR frame (for kernels)
        prev_detections: Previous detections (for kernels)
        kernel_cache: KernelCache object
        args: Arguments object with tracking parameters
        use_gpu: Whether to use GPU acceleration. Without CuPy the numpy path
            scores the same pair set (every pair inside the appearance gate when
            the graph-struct matcher is active), so CPU and GPU give the same
            matrix up to float rounding.
        frame_lab_gpu: Optional LAB frame already on the GPU.
        predicted_positions: Optional (M, 2) motion-prior positions used as the
            track centres for the distance gate (None = current positions).
        neighbor_counts: Optional per-track neighbour counts for the adaptive
            weights.
        graph_struct_data: Precomputed structural-matcher data
            (``match_ops.build_graph_struct_data``) for ``--match-op graph-struct``.
        dump_frame: Processed-frame index; with env ``CBTV_DUMP_C`` set, the
            final matrix is written to ``C_<frame>.npz`` (None = never).

    Returns:
        C: Cost matrix (M, N) with np.float32
        kernel_info: Dict with kernel computation times for profiling
    """
    import time as time_module

    M = tracks_soa.count
    N = len(detections)

    kernel_info = {
        't_dist': 0.0,
        't_color': 0.0,
        't_kernel_build': 0.0,
        't_kernel_corr': 0.0,
        'kernel_hits': 0,
        'kernel_misses': 0,
        # [Phase 0 V0 instrumentation] Gate-survival + kernel-eval counters.
        # n_pairs_total       = M * N (full unfiltered cost-matrix size)
        # n_pairs_gated_dist  = #pairs surviving the spatial distance gate (max_dist)
        # n_pairs_gated_combined = #pairs surviving distance AND colour gate (= |valid_pairs|)
        # n_kernel_evals      = #pairs for which kernel NCC was actually computed
        # mean_cands_per_track = n_pairs_gated_combined / max(1, M)
        'n_pairs_total': M * N,
        'n_pairs_gated_dist': 0,
        'n_pairs_gated_combined': 0,
        'n_kernel_evals': 0,
        'mean_cands_per_track': 0.0,
    }

    if M == 0 or N == 0:
        C = np.zeros((M, N), np.float32)
        _maybe_dump_C(dump_frame, C, tracks_soa, N, None, detections)
        return C, kernel_info

    # Initialize cost matrix with large values
    C = np.full((M, N), 1e6, np.float32)

    # [COST-1] Vectorized distance matrix (GPU)
    # Phase 2: if predicted_positions (M, 2) was supplied (computed by the
    # affine field in run_tracker.py), use it as the per-track "current
    # centre" for the distance gate so the candidate set follows the
    # predicted motion.  predicted_positions=None preserves the Phase-1
    # behavior exactly.
    t0 = time_module.perf_counter()
    det_positions = np.array([[x, y] for (x, y, _, _) in detections], dtype=np.float32)
    if predicted_positions is not None:
        _track_pos_for_gate = np.asarray(predicted_positions, dtype=np.float32)
    else:
        _track_pos_for_gate = tracks_soa.positions
    dist_matrix = vectorized_distance_matrix_gpu(_track_pos_for_gate, det_positions, use_gpu=use_gpu)
    kernel_info['t_dist'] = (time_module.perf_counter() - t0) * 1000.0

    # [COST-2] Distance hard gate
    dist_gate = (dist_matrix <= args.max_dist)
    # [Phase 0 V0] Count distance-gate survivors.
    kernel_info['n_pairs_gated_dist'] = int(dist_gate.sum())

    # [COST-3] Batch color distance (GPU) - only for pairs passing distance gate
    t0 = time_module.perf_counter()
    track_dets = tracks_soa.get_detections_format()
    # [--color-source track] compare detections with the colour each track
    # stored when it was last matched, instead of re-sampling the CURRENT
    # frame at the track's previous position (which is mostly background once
    # a particle moves more than ~r/2 per frame). Default 'frame' = legacy.
    # 'min' takes, per pair, the smaller of the two distances: the frame
    # re-sample wins for slow particles (same-frame lighting cancels), the
    # stored colour wins once the old position no longer covers the particle.
    _csrc = getattr(args, 'color_source', 'frame')
    _track_lab_S = None
    if _csrc in ('track', 'min') and getattr(tracks_soa, 'lab_w', None) is not None:
        _track_lab_S = (tracks_soa.lab_w, tracks_soa.S_w)
    _cargs = dict(saturation_weight=args.color_saturation_weight,
                  nonlinear_k=args.color_nonlinear_k,
                  glare_threshold=args.glare_threshold,
                  use_gpu=(use_gpu and args.opt_gpu_color == 1))
    color_dist_matrix, S_conf_matrix = batch_smart_color_distance_gpu(
        tracks_soa.colors, np.array(det_colors, dtype=np.float32),
        frame_bgr, track_dets, detections,
        track_lab_S=(_track_lab_S if _csrc == 'track' else None), **_cargs)
    if _csrc == 'min' and _track_lab_S is not None:
        _cd2, _sc2 = batch_smart_color_distance_gpu(
            tracks_soa.colors, np.array(det_colors, dtype=np.float32),
            frame_bgr, track_dets, detections, track_lab_S=_track_lab_S, **_cargs)
        _use2 = _cd2 < color_dist_matrix
        color_dist_matrix = np.where(_use2, _cd2, color_dist_matrix)
        S_conf_matrix = np.where(_use2, _sc2, S_conf_matrix)
    kernel_info['t_color'] = (time_module.perf_counter() - t0) * 1000.0

    # [COST-4] Color hard gate (widened: higher S_conf threshold + wider distance)
    color_gate = ~((S_conf_matrix > 0.5) & (color_dist_matrix > args.max_color * 1.2))

    # Combined gate: valid_pairs is a boolean matrix of shape (M, N)
    valid_pairs = dist_gate & color_gate
    # [Phase 0 V0] Count combined-gate survivors + mean-candidates-per-track.
    kernel_info['n_pairs_gated_combined'] = int(valid_pairs.sum())
    kernel_info['mean_cands_per_track'] = float(kernel_info['n_pairs_gated_combined']) / float(max(1, M))

    # [COST-5] Gather-Correlate-Scatter kernel NCC (GPU)
    t0_corr_total = time.perf_counter()

    if use_gpu and cp is not None and frame_lab_gpu is not None:
        # Move required matrices to GPU
        valid_pairs_gpu = cp.asarray(valid_pairs)
        dist_matrix_gpu = cp.asarray(dist_matrix)
        color_dist_matrix_gpu = cp.asarray(color_dist_matrix)
        S_conf_matrix_gpu = cp.asarray(S_conf_matrix)

        # Initialize GPU cost matrix
        C_gpu = cp.full((M, N), 1e6, dtype=cp.float32)
        # ---------------------------------------------------------
        #  Phase 2: Gather-Correlate-Scatter
        # ---------------------------------------------------------
        t0_corr_total = time_module.perf_counter()

        if use_gpu and cp is not None and frame_lab_gpu is not None:
            # Move required matrices to GPU
            # valid_pairs_gpu = cp.asarray(valid_pairs) # This is now computed differently
            dist_matrix_gpu = cp.asarray(dist_matrix)
            color_dist_matrix_gpu = cp.asarray(color_dist_matrix)
            S_conf_matrix_gpu = cp.asarray(S_conf_matrix)

            # Initialize GPU cost matrix
            C_gpu = cp.full((M, N), 1e6, dtype=cp.float32)

            # ---------------------------------------------------------
            #  Phase 2: Gather
            # ---------------------------------------------------------
            # Find all (track_idx, det_idx) pairs that need correlation
            if args.w_dist + args.w_color < 1.0: # Only perform kernel correlation if gamma has weight
                # Step 1: Identify Valid Pairs for kernel correlation
                # We use a slightly larger distance gate here to allow for some flexibility
                # in kernel search, as the kernel might be slightly off-center.
                # The final cost will still be gated by max_dist.
                # We use a slightly larger distance gate here to allow for some flexibility
                # in kernel search, as the kernel might be slightly off-center.
                # The final cost will still be gated by max_dist.
                # [Phase 5 Stage 3] Shared appearance gate in diameters (match_gate_d>0)
                # used by BOTH operators; 0 = legacy (kernel: 2*max_dist).
                _gate_d = float(getattr(args, 'match_gate_d', 0.0))
                _appear_gate_px = (_gate_d * 2.0 * float(args.r_max)) if _gate_d > 0.0 else (args.max_dist * 2.0)
                valid_pairs_for_kernel_gpu = dist_matrix_gpu < _appear_gate_px

                # Extract lists of valid indices
                i_indices_cp, j_indices_cp = cp.where(valid_pairs_for_kernel_gpu)
                i_indices = i_indices_cp.get()
                j_indices = j_indices_cp.get()

                B = len(i_indices)
                # [Phase 0 V0] Count kernel NCC evaluations actually performed this frame.
                kernel_info['n_kernel_evals'] = int(B)

                # Compute fallback costs dynamically on GPU for all valid pairs
                w_d_base = args.w_dist / (args.w_dist + args.w_color) if (args.w_dist + args.w_color) > 0 else 0.0
                w_c_base = args.w_color / (args.w_dist + args.w_color) if (args.w_dist + args.w_color) > 0 else 0.0

                fallback_costs_gpu = w_d_base * dist_matrix_gpu + w_c_base * color_dist_matrix_gpu
                C_gpu = cp.where(valid_pairs_gpu, fallback_costs_gpu, C_gpu)

                H, W = frame_lab.shape[:2]

                if B > 0:
                    patch_size = 21
                    half_p = patch_size // 2
                    # 1. GATHER KERNELS (Two-pass approach to determine patch_size dynamically)
                    unique_i_indices = np.unique(i_indices)
                    num_unique_i = len(unique_i_indices)

                    # First pass: find the maximum kernel size (khalf)
                    max_khalf_cpu = 0
                    K_metadata = [] # Store (kh, kw, K) to avoid refetching
                    K_khalfs = np.zeros(num_unique_i, dtype=int) # Store khalf for each unique kernel

                    for idx, i in enumerate(unique_i_indices):
                        t_id = int(tracks_soa.ids[i])
                        k_data = kernel_cache.get(t_id)
                        if k_data is not None:
                            K, _, _ = k_data
                            kh, kw = K.shape[:2]
                            current_khalf = kh // 2
                            max_khalf_cpu = max(max_khalf_cpu, current_khalf)
                            K_metadata.append((True, kh, kw, K))
                            K_khalfs[idx] = current_khalf
                        else:
                            K_metadata.append((False, 0, 0, None))
                            K_khalfs[idx] = 0 # No kernel, so khalf is 0

                    # Set patch half-size dynamically based on max kernel size
                    # If no kernels are found, default to a reasonable size (e.g., 10 for 21x21 patch)
                    half_p = max_khalf_cpu if max_khalf_cpu > 0 else 10
                    patch_size = 2 * half_p + 1

                    # Pre-allocate arrays for unique kernels and their properties
                    K_unique_cpu = np.zeros((num_unique_i, 3, patch_size, patch_size), dtype=np.float32)
                    valid_unique_mask = np.zeros(num_unique_i, dtype=bool)
                    i_to_unique_idx_map = np.zeros(M, dtype=int) # Map track index to unique kernel index

                    # Second pass: Populate unique kernels and their properties
                    for idx, i in enumerate(unique_i_indices):
                        i_to_unique_idx_map[i] = idx
                        valid, kh, kw, K = K_metadata[idx]
                        if valid:
                            # Calculate padding and cropping for kernel extraction
                            y0, y1 = max(0, kh//2 - half_p), min(kh, kh//2 + half_p + 1)
                            x0, x1 = max(0, kw//2 - half_p), min(kw, kw//2 + half_p + 1)

                            ty0 = half_p - (kh//2 - y0)
                            ty1 = ty0 + (y1 - y0)
                            tx0 = half_p - (kw//2 - x0)
                            tx1 = tx0 + (x1 - x0)

                            K_unique_cpu[idx, :, ty0:ty1, tx0:tx1] = K[y0:y1, x0:x1, :].transpose(2, 0, 1)
                            valid_unique_mask[idx] = True

                    # Move data to GPU
                    K_unique_gpu = cp.asarray(K_unique_cpu)
                    valid_unique_mask_gpu = cp.asarray(valid_unique_mask)
                    i_to_unique_idx_gpu = cp.asarray(i_to_unique_idx_map)
                    K_khalfs_gpu = cp.asarray(K_khalfs)

                    # 2. GATHER ALL UNIQUE DETECTIONS ONCE
                    cx_all = np.array([det[0] for det in detections])
                    cy_all = np.array([det[1] for det in detections])
                    cx_gpu_all = cp.asarray(cx_all)
                    cy_gpu_all = cp.asarray(cy_all)
                    N_total = len(detections)

                    dy, dx = cp.meshgrid(cp.arange(-half_p, half_p+1), cp.arange(-half_p, half_p+1), indexing='ij')
                    sample_y = cp.clip(cp.round(cy_gpu_all)[:, None, None] + dy[None, :, :], 0, H-1).astype(cp.int32)
                    sample_x = cp.clip(cp.round(cx_gpu_all)[:, None, None] + dx[None, :, :], 0, W-1).astype(cp.int32)

                    ch_idx = cp.arange(3)[None, :, None, None]
                    frame_lab_ch = cp.transpose(frame_lab_gpu, (2, 0, 1))
                    det_patches_all = frame_lab_ch[ch_idx, sample_y[:, None, :, :], sample_x[:, None, :, :]] # (N_total, 3, 21, 21)

                    # 3. DENSE CORRELATE (GEMM) - Grouped by Kernel Size
                    cc_scores = cp.zeros(B, dtype=cp.float32)
                    b_unique_indices_gpu = i_to_unique_idx_gpu[i_indices_cp]
                    valid_b_mask = valid_unique_mask_gpu[b_unique_indices_gpu]

                    # Grouped GEMM for Correlation
                    cc_scores = cp.zeros(B, dtype=cp.float32)
                    b_unique_indices_gpu = i_to_unique_idx_gpu[i_indices_cp]
                    valid_b_mask = valid_unique_mask_gpu[b_unique_indices_gpu]

                    # Find the range of khalf values present
                    if B > 0 and num_unique_i > 0:
                        min_khalf = int(cp.min(K_khalfs_gpu))
                        max_khalf = int(cp.max(K_khalfs_gpu))

                        for khalf in range(min_khalf, max_khalf + 1):
                            size_mask = (K_khalfs_gpu == khalf)
                            if not cp.any(size_mask): continue

                            b_mask = size_mask[b_unique_indices_gpu]
                            if not cp.any(b_mask): continue

                            c0 = half_p - khalf
                            c1 = half_p + khalf + 1
                            kh_size = c1 - c0

                            idx_of_this_size = cp.where(size_mask)[0]
                            K_sub = K_unique_gpu[idx_of_this_size, :, c0:c1, c0:c1]

                            det_crops_all = det_patches_all[:, :, c0:c1, c0:c1]

                            det_mean = cp.mean(det_crops_all, axis=(2, 3), keepdims=True)
                            det_std = cp.std(det_crops_all, axis=(2, 3), keepdims=True) + 1e-8
                            det_crops_norm = ((det_crops_all - det_mean) / det_std).astype(cp.float32)

                            K_flat = K_sub.reshape(len(idx_of_this_size), -1).astype(cp.float32)
                            det_flat = det_crops_norm.reshape(N_total, -1).astype(cp.float32)

                            dense_sub = cp.dot(K_flat, det_flat.T) / (3 * kh_size * kh_size)

                            # We need to map `b_unique_indices_gpu[b_mask]` which are values in `[0, num_unique_i-1]`
                            # to their position in `idx_of_this_size`.
                            # Create a reverse mapping for this size
                            reverse_map = cp.zeros(num_unique_i, dtype=cp.int32)
                            reverse_map[idx_of_this_size] = cp.arange(len(idx_of_this_size), dtype=cp.int32)

                            mapped_b_idx = reverse_map[b_unique_indices_gpu[b_mask]]
                            j_idx = j_indices_cp[b_mask]

                            cc_scores[b_mask] = dense_sub[mapped_b_idx, j_idx]

                    # [Phase 3 Stage A] Removed a dead `cc_scores.get()` here:
                    # the CPU copy was never used and forced a GPU sync per
                    # frame inside the timed kernel branch.

                    # 4. SCATTER
                    S_conf_batch_gpu = S_conf_matrix_gpu[i_indices_cp, j_indices_cp]
                    dist_batch_gpu = dist_matrix_gpu[i_indices_cp, j_indices_cp]
                    color_dist_batch_gpu = color_dist_matrix_gpu[i_indices_cp, j_indices_cp]

                    # Vectorized compute_adaptive_weights_normalized logic —
                    # extracted to adaptive_weight_terms() (Phase 3 Stage A).
                    # [Wart fix] neigh_factor now uses the per-pair track
                    # neighbour counts when the caller supplies them, instead
                    # of the constant args.min_neighbors.
                    if neighbor_counts is not None:
                        _nc_batch = cp.asarray(neighbor_counts, dtype=cp.float32)[i_indices_cp]
                    else:
                        _nc_batch = None
                    w_dist_final, w_color_effective, gamma_final = adaptive_weight_terms(
                        args.w_dist, args.w_color, args.gamma,
                        S_conf_batch_gpu, _nc_batch,
                        min_neighbors_legacy=args.min_neighbors, xp=cp,
                    )

                    # [Phase 5 Stage 3] --match-op graph-struct: replace the
                    # kernel-NCC appearance score with the kernel-free structural
                    # score per gated pair (the kernel build was skipped upstream,
                    # so cc_scores is empty here). Default 'kernel' leaves cc_scores
                    # untouched -> byte-identical.
                    if getattr(args, 'match_op', 'kernel') == 'graph-struct' and graph_struct_data is not None:
                        from .match_ops import graph_struct_cc_for_pairs
                        # graph-struct neighbour gate: match_gate_d*diameter if set,
                        # else the --graph-struct-gate px (default 22).
                        _gs_gate = (_gate_d * 2.0 * float(args.r_max)) if _gate_d > 0.0 \
                            else float(getattr(args, 'graph_struct_gate', 22.0))
                        _gs_sc, _gs_va = graph_struct_cc_for_pairs(
                            i_indices, j_indices, det_positions,
                            np.asarray(det_colors, dtype=np.float64), graph_struct_data,
                            gate_px=_gs_gate, colour_max=float(args.max_color))
                        cc_scores = cp.asarray(_gs_sc, dtype=cp.float32)
                        valid_b_mask = cp.asarray(_gs_va)

                    sNCC = cc_scores
                    cost_batch = w_dist_final * dist_batch_gpu + w_color_effective * color_dist_batch_gpu + gamma_final * (1.0 - sNCC)

                    # Handle invalid kernels with fallback costs
                    cost_batch = cp.where(valid_b_mask, cost_batch, fallback_costs_gpu[i_indices_cp, j_indices_cp])

                    C_gpu[i_indices_cp, j_indices_cp] = cost_batch
                    C = C_gpu.get()

                else:
                    C = C_gpu.get()

            # If no kernel correlation was performed (e.g., B=0 or w_dist+w_color >= 1.0),
            # ensure C is initialized with fallback costs for all valid pairs.
            # This handles cases where gamma is not used or no valid kernel pairs exist.
            if args.w_dist + args.w_color >= 1.0 or B == 0:
                w_d_base = args.w_dist / (args.w_dist + args.w_color) if (args.w_dist + args.w_color) > 0 else 0
                w_c_base = args.w_color / (args.w_dist + args.w_color) if (args.w_dist + args.w_color) > 0 else 0

                fallback_costs_gpu = w_d_base * dist_matrix_gpu + w_c_base * color_dist_matrix_gpu
                C_gpu = cp.where(valid_pairs_gpu, fallback_costs_gpu, C_gpu)
                C = C_gpu.get()

            kernel_info['t_kernel_corr'] = (time_module.perf_counter() - t0_corr_total) * 1000.0

            # [COST-5 cont.] [Phase 5-D] Assignment persistence bonus
            # Prefer continuing previous frame's assignment to reduce ID oscillation
            persistence_bonus = getattr(args, 'persistence_bonus', 0.85)
            if persistence_bonus < 1.0 and M > 0 and N > 0:
                prev_pos_arr = tracks_soa.last_det_pos  # (M, 2) array, invalid = [-1, -1]
                valid = prev_pos_arr[:, 0] > 0  # (M,) boolean mask
                if np.any(valid):
                    from scipy.spatial import cKDTree
                    det_positions_kd = np.array([[x, y] for (x, y, _, _) in detections], dtype=np.float32)
                    det_tree = cKDTree(det_positions_kd)
                    dists_prev, indices_prev = det_tree.query(prev_pos_arr[valid])
                    radius_thresh = tracks_soa.radii[valid] * 1.5
                    close_enough = dists_prev < radius_thresh
                    valid_idx = np.where(valid)[0]
                    for k in range(len(valid_idx)):
                        if close_enough[k]:
                            C[valid_idx[k], indices_prev[k]] *= persistence_bonus

            _maybe_dump_C(dump_frame, C, tracks_soa, N, valid_pairs, detections)
            return C, kernel_info

    # ------------------------------------------------------------------
    # [Standalone CPU fallback] The Gather-Correlate-Scatter kernel-NCC
    # path above is GPU-only; without it the function used to fall off the
    # end and return None on CPU. This branch composes the cost matrix on
    # numpy: the graph-struct structural score (pure numpy, identical to
    # the GPU path's) supplies the appearance term when --match-op
    # graph-struct is active; for the legacy 'kernel' operator the
    # appearance term is skipped and dist+colour weights renormalise.
    # ------------------------------------------------------------------
    # GPU parity: the GPU branch (w_dist + w_color < 1) scores EVERY pair
    # inside the appearance gate, overwriting the dist+colour-gated
    # fallback; pairs that fail the gates keep the ungated fallback cost.
    _gpu_like = (getattr(args, 'match_op', 'kernel') == 'graph-struct'
                 and graph_struct_data is not None and (args.w_dist + args.w_color) < 1.0)
    if _gpu_like:
        wsum0 = max(1e-9, args.w_dist + args.w_color)
        _fb_all = ((args.w_dist / wsum0) * dist_matrix + (args.w_color / wsum0) * color_dist_matrix)
        C = np.where(valid_pairs, _fb_all, C).astype(np.float32)
        _gd = float(getattr(args, 'match_gate_d', 0.0))
        _appear = (_gd * 2.0 * float(args.r_max)) if _gd > 0.0 else (args.max_dist * 2.0)
        i_idx, j_idx = np.where(dist_matrix < _appear)
    else:
        i_idx, j_idx = np.where(valid_pairs)
    if len(i_idx):
        S_conf_batch = S_conf_matrix[i_idx, j_idx]
        dist_batch = dist_matrix[i_idx, j_idx]
        color_batch = color_dist_matrix[i_idx, j_idx]
        _nc_batch = (np.asarray(neighbor_counts, dtype=np.float32)[i_idx]
                     if neighbor_counts is not None else None)
        w_dist_final, w_color_eff, gamma_final = adaptive_weight_terms(
            args.w_dist, args.w_color, args.gamma, S_conf_batch, _nc_batch,
            min_neighbors_legacy=args.min_neighbors, xp=np)
        wsum = max(1e-9, args.w_dist + args.w_color)
        fb = ((args.w_dist / wsum) * dist_batch
              + (args.w_color / wsum) * color_batch)
        use_gs = (getattr(args, 'match_op', 'kernel') == 'graph-struct'
                  and graph_struct_data is not None
                  and (args.w_dist + args.w_color) < 1.0)
        if use_gs:
            from .match_ops import graph_struct_cc_for_pairs
            _gate_d = float(getattr(args, 'match_gate_d', 0.0))
            _gs_gate = (_gate_d * 2.0 * float(args.r_max)) if _gate_d > 0.0 \
                else float(getattr(args, 'graph_struct_gate', 22.0))
            _gs_sc, _gs_va = graph_struct_cc_for_pairs(
                i_idx, j_idx, det_positions,
                np.asarray(det_colors, dtype=np.float64), graph_struct_data,
                gate_px=_gs_gate, colour_max=float(args.max_color))
            cost_batch = (w_dist_final * dist_batch
                          + w_color_eff * color_batch
                          + gamma_final * (1.0 - np.asarray(_gs_sc, np.float32)))
            cost_batch = np.where(np.asarray(_gs_va, bool), cost_batch, fb)
        else:
            cost_batch = fb
        C[i_idx, j_idx] = cost_batch.astype(np.float32)
    kernel_info['t_kernel_corr'] = (time_module.perf_counter() - t0_corr_total) * 1000.0

    persistence_bonus = getattr(args, 'persistence_bonus', 0.85)
    if persistence_bonus < 1.0 and M > 0 and N > 0:
        prev_pos_arr = tracks_soa.last_det_pos
        valid = prev_pos_arr[:, 0] > 0
        if np.any(valid):
            from scipy.spatial import cKDTree
            det_tree = cKDTree(det_positions)
            dists_prev, indices_prev = det_tree.query(prev_pos_arr[valid])
            radius_thresh = tracks_soa.radii[valid] * 1.5
            close_enough = dists_prev < radius_thresh
            valid_idx = np.where(valid)[0]
            for k in range(len(valid_idx)):
                if close_enough[k]:
                    C[valid_idx[k], indices_prev[k]] *= persistence_bonus

    _maybe_dump_C(dump_frame, C, tracks_soa, N, valid_pairs, detections)
    return C, kernel_info

