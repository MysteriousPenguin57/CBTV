"""Detection pipeline orchestration (exemplars -> NCC -> Bayes)."""
import math
import os

import cv2
import numpy as np

from . import gpu
from .bayes import improved_bayesian_refinement_temporal
from .correlation import build_channels, multiscale_corr
from .detection_filter import peaks_from_map
from .exemplars import (
    hough_exemplars_dual,
    intelligent_nms_radius,
    multiframe_hough_exemplars,
    norm_kernel_from_exemplars,
    propose_exemplars_fallback,
    score_channel,
)
from .gpu import cp, to_gpu, to_numpy
from .log_detect import blob_log_gpu_fast


def user_detection_with_bayes_adapt(
        bgr, rmin, rmax, thr_abs, topk=3, nms_factor=0.6, hough_polarity="auto",
        refine_bayes=1, refine_top_pos=300, refine_top_neg=300, refine_beta=0.6, scales=(0.9,1.0,1.1),
        thr_mode="hybrid", thr_local_k=2.5, thr_local_ksize=0, thr_topk_tile=0, thr_tile_factor=3.0,
        dedup_threshold=3.0, intelligent_nms=True, dump=None,
        frame_buffer_gray=None, current_frame_idx=0, multiframe_offset=5, fp_memory=None,
        enable_subpixel=False, use_antialias=False, use_simpson=False,
        opt_gpu_blob=1, force_log=False, obstacle_mask_bool=None, rng=None):
    """
    Detection with optional multi-frame candidate pooling

    Args:
        frame_buffer_gray: Optional list of gray frames for multi-frame detection
        current_frame_idx: Index of current frame in buffer
        multiframe_offset: Frame offset for multi-frame detection (default 5)
        obstacle_mask_bool: Optional boolean mask (True=valid, False=suppress)
    """
    import time
    t0_all = time.perf_counter()
    chans, gray = build_channels(bgr)
    t_chans = time.perf_counter() - t0_all

    # [DET-1] Hough exemplar generation + force_log bypass (see Flowchart 6 — LoG Detection)
    # [v6 Change] force_log (Phase 4B): small-image mode skips Hough entirely because
    # HoughCircles requires minimum resolution (~256px) to detect circle arcs reliably.
    hough_ex = []
    t1 = time.perf_counter()
    if force_log:
        # Small-image mode: skip Hough entirely, use LoG directly
        hough_ex = []
        if current_frame_idx == 0:
            print("[SMALL-IMAGE] Skipping Hough — force_log=True")
    elif frame_buffer_gray is not None and len(frame_buffer_gray) > 1:
        hough_ex = multiframe_hough_exemplars(
            frame_buffer_gray, current_frame_idx,
            rmin=rmin, rmax=rmax, param2=12,
            minDist=max(6, rmin-2), nmax=800,
            polarity=hough_polarity, offset=multiframe_offset
        )
    else:
        # Single-frame detection (original behavior)
        hough_ex = hough_exemplars_dual(chans["Gray"], rmin=rmin, rmax=rmax, param2=12, minDist=max(6, rmin-2), nmax=800, polarity=hough_polarity)
    t_hough = time.perf_counter() - t1

    # [DET-2] LoG supplement/fallback decision tree (530/400 thresholds)
    # [Phase 3] Adaptive LoG Fallback: If Hough struggles, use LoG to supplement or replace.
    # [Decision Rationale] (see Flowchart 6 — LoG Detection) The primary video has 534 particles. Hough reliably finds ~530+ in
    # normal frames but degrades in dense overlap regions where circles merge. When Hough
    # produces <530 exemplars, the missing ones tend to be in dense clusters — precisely where
    # blurry Hough kernels "poison" the NCC correlation step downstream.
    # - 530 (supplement): ~534 particles minus small margin; below this, LoG fills gaps
    # - 400 (replace): severe Hough failure; ALL Hough exemplars may be poisoned, so discard
    #   them entirely and rely on LoG which handles dense regions better via scale-space.
    # LoG params: sigma 3.5-6.5 covers particle radius range (~6-9px), threshold=0.05 tuned
    # to detect ~90% of particles while rejecting most noise blobs.
    ex = hough_ex
    log_ex = None
    t1 = time.perf_counter()
    LOG_REPLACE_THRESHOLD = 400   # Below this: LoG replaces Hough entirely
    LOG_SUPPLEMENT_THRESHOLD = 530  # Below this: LoG supplements Hough
    if force_log or len(hough_ex) < LOG_SUPPLEMENT_THRESHOLD:
        # LoG-based center finding (GPU FAST if enabled)
        log_input = chans["Gray"].copy()
        if current_frame_idx == 0:
            print(f"[DIAG] USE_CUPY={gpu.USE_CUPY} opt_gpu_blob={opt_gpu_blob}")
        if current_frame_idx == 0:
            print(f"[DIAG] log_input dtype={log_input.dtype} flags={log_input.flags.c_contiguous} max={log_input.max()}")
        if force_log:
            # Small-image mode: use adjusted sigma range for potentially smaller particles
            small_min_sigma = max(1.0, rmin * 0.4)
            small_max_sigma = max(3.5, rmax * 0.6)
            # Normalize contrast for low-intensity images
            img_min, img_max = float(log_input.min()), float(log_input.max())
            if img_max - img_min < 10:
                # Nearly flat image — no detectable features
                blobs = np.zeros((0, 4))
                if current_frame_idx < 5 or current_frame_idx % 50 == 0:
                    print(f"[SMALL-IMAGE] Frame {current_frame_idx}: flat image (range {img_min:.0f}-{img_max:.0f}), skipping LoG")
            else:
                # Stretch contrast to use full 0-255 range
                log_input_norm = np.clip((log_input.astype(np.float32) - img_min) / (img_max - img_min) * 255.0, 0, 255).astype(np.uint8)
                blobs = blob_log_gpu_fast(log_input_norm, min_sigma=small_min_sigma, max_sigma=small_max_sigma,
                                           num_sigma=6, threshold=0.02, use_gpu=(opt_gpu_blob == 1))
                if current_frame_idx < 5 or current_frame_idx % 50 == 0:
                    print(f"[SMALL-IMAGE] Frame {current_frame_idx}: contrast-normalized ({img_min:.0f}-{img_max:.0f}->0-255), "
                          f"sigma={small_min_sigma:.1f}-{small_max_sigma:.1f}, {len(blobs)} blobs")
        else:
            blobs = blob_log_gpu_fast(log_input, min_sigma=3.5, max_sigma=6.5, num_sigma=4, threshold=0.05, use_gpu=(opt_gpu_blob == 1))

        r_guess = (rmin + rmax) / 2.0
        log_ex = []
        for blob in blobs:
            if len(blob) == 3:
                y, x, sigma = blob
                r_est = sigma * 1.414
                if r_est < rmin:
                    r_est = r_guess
                log_ex.append((float(x), float(y), float(r_est)))

        print(f"[DIAG] opt_gpu_blob={opt_gpu_blob} len(log_ex)={len(log_ex)} len(hough_ex)={len(hough_ex)} mode={'replace' if len(hough_ex) < LOG_REPLACE_THRESHOLD else 'supplement'}")

        # [DET-3] cKDTree deduplication merge (LoG + Hough at 1.2× radius threshold)
        if log_ex:
            if len(hough_ex) < LOG_REPLACE_THRESHOLD:
                # Severe Hough failure: LoG replaces entirely (original behavior)
                ex = log_ex
            else:
                # Moderate deficit: merge LoG + Hough, deduplicate
                from scipy.spatial import cKDTree
                hough_coords = np.array([[x, y] for x, y, r in hough_ex])
                merged_ex = list(hough_ex)
                if len(hough_coords) > 0:
                    tree = cKDTree(hough_coords)
                    for lx, ly, lr in log_ex:
                        dist, _ = tree.query([lx, ly])
                        if dist > lr * 1.2:  # LoG detection not near any Hough detection
                            merged_ex.append((lx, ly, lr))
                else:
                    merged_ex = log_ex
                print(f"[DIAG] LoG supplement: {len(hough_ex)} hough + {len(merged_ex) - len(hough_ex)} new LoG = {len(merged_ex)} total")
                ex = merged_ex
    t_log = time.perf_counter() - t1

    if not ex:
        r_guess = int((rmin + rmax)//2)
        ex = propose_exemplars_fallback(chans["Gray"], r_guess=r_guess, nmax=400, win=max(5, int(0.8*r_guess)))
    if not ex:
        return [], [], gray, {"reason":"no_exemplars"}

    # [DET-4] Exemplar scoring, kernel building, and NCC correlation pipeline
    radii = [r for _,_,r in ex]
    if gpu.USE_CUPY:
        rad = int(cp.median(cp.array(radii)))
    else:
        rad = int(np.median(radii))

    t1 = time.perf_counter()
    results = {}
    for name,chan in chans.items():
        try:
            k, r_est = norm_kernel_from_exemplars(chan, ex, pad=0, rad=rad)
            s = score_channel(chan.astype(np.float32), ex, k, r_est, holdout=0.3, rng=rng)
            results[name] = dict(kernel=k, score=s, rad=r_est, chan=chan.astype(np.float32))
        except Exception:
            continue
    t_score = time.perf_counter() - t1

    if not results:
        return [], [], gray, {"reason":"no_kernels"}

    ranked = sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)
    K = max(1, min(topk, len(ranked)))
    names = [name for name,_ in ranked[:K]]
    scores = [results[n]["score"] for n in names]

    if gpu.USE_CUPY:
        scores_gpu = cp.array(scores, dtype=cp.float32)
        sc_max = float(scores_gpu.max())
        top_sc = float(scores_gpu[0])
        weights = cp.exp(scores_gpu - sc_max) / (cp.exp(top_sc - sc_max) + 1e-6)
        weights = to_numpy(weights)
    else:
        scores = np.array(scores, np.float32)
        sc_max = float(scores.max())
        top_sc = float(scores[0])
        weights = np.exp(scores - sc_max) / (np.exp(top_sc - sc_max) + 1e-6)

    rad_vals = [results[n]["rad"] for n in names]
    if gpu.USE_CUPY:
        rad_use = int(cp.median(cp.array(rad_vals)))
    else:
        rad_use = int(np.median(rad_vals))

    t1 = time.perf_counter()
    corr_maps = {}
    for n in names:
        corr_maps[n] = multiscale_corr(results[n]["chan"], results[n]["kernel"], scales=scales,
                                        use_antialias=use_antialias, use_simpson=use_simpson)
    t_corr = time.perf_counter() - t1

    t1 = time.perf_counter()
    fused = None
    for wi, n in zip(weights, names, strict=False):
        m = corr_maps[n] * float(wi)
        fused = m if fused is None else np.maximum(fused, m)
    t_fuse = time.perf_counter() - t1

    # [Phase 5-G2] Apply obstacle mask to fused correlation map (pre-peak zeroing)
    if obstacle_mask_bool is not None and fused is not None:
        h_f, w_f = fused.shape[:2]
        h_m, w_m = obstacle_mask_bool.shape[:2]
        if (h_f, w_f) == (h_m, w_m):
            fused[~obstacle_mask_bool] = 0.0
        else:
            mask_resized = cv2.resize(obstacle_mask_bool.astype(np.uint8), (w_f, h_f),
                                       interpolation=cv2.INTER_NEAREST) > 0
            fused[~mask_resized] = 0.0

    is_log_fallback = (ex is log_ex) and (log_ex is not None)

    effective_thr_abs = thr_abs
    if is_log_fallback:
        effective_thr_abs = min(thr_abs, 0.35)

    if intelligent_nms:
        nms_rad = intelligent_nms_radius(ex, base_factor=nms_factor)
    else:
        nms_rad = max(4, int(nms_factor * rad_use))

    t1 = time.perf_counter()
    lk = thr_local_k
    lksz = int(thr_local_ksize) if thr_local_ksize and thr_local_ksize>0 else None
    dets_raw = peaks_from_map(fused, rad_use, effective_thr_abs, nms_rad, mode=thr_mode, local_k=lk, local_ksize=lksz,
                              topk_per_tile=int(thr_topk_tile), tile_factor=float(thr_tile_factor),
                              enable_subpixel=enable_subpixel)
    t_peaks = time.perf_counter() - t1

    print(f"[DIAG] mode={thr_mode} thr={effective_thr_abs} len(dets_raw)={len(dets_raw)}")

    t1 = time.perf_counter()
    dets_bayes_added = []
    if refine_bayes and len(dets_raw)>=8:
        if fp_memory is not None:
            dets_bayes_added = improved_bayesian_refinement_temporal(
                chans, gray, fused, dets_raw, rad_use, names, weights, results,
                fp_memory, refine_top_pos=int(refine_top_pos), refine_beta=float(refine_beta),
                scales=scales, thr_abs=effective_thr_abs, thr_mode=thr_mode, lk=lk, lksz=lksz,
                thr_topk_tile=thr_topk_tile, thr_tile_factor=thr_tile_factor, nms_rad=nms_rad
            )
        else:
            # Original Bayesian refinement
            pos = sorted(dets_raw, key=lambda d: d[3], reverse=True)[:int(refine_top_pos)]
            neg = sorted(dets_raw, key=lambda d: d[3])[:int(refine_top_neg//2)]

            if gpu.USE_CUPY:
                fused_gpu = to_gpu(fused)
                corr_min = float(cp.percentile(fused_gpu, 10.0))
            else:
                corr_min = float(np.percentile(fused, 10.0))

            H,W = fused.shape[:2]
            step = max(4, int(0.7*rad_use))
            extra_neg=[]
            for y in range(rad_use, H-rad_use, step):
                for x in range(rad_use, W-rad_use, step):
                    if fused[y,x] <= corr_min:
                        extra_neg.append((float(x), float(y), float(rad_use), float(fused[y,x])))
                        if len(extra_neg) >= int(refine_top_neg//2): break
                if len(extra_neg) >= int(refine_top_neg//2): break
            neg += extra_neg

            pos_radii = [r for _,_,r,_ in pos]
            neg_radii = [r for _,_,r,_ in neg]
            if gpu.USE_CUPY:
                pos_rad = int(cp.median(cp.array(pos_radii))) if pos else rad_use
                neg_rad = int(cp.median(cp.array(neg_radii))) if neg else rad_use
            else:
                pos_rad = int(np.median(pos_radii)) if pos else rad_use
                neg_rad = int(np.median(neg_radii)) if neg else rad_use

            pos_kernels={}; neg_kernels={}
            for n in names:
                pos_kernels[n], _ = norm_kernel_from_exemplars(results[n]["chan"], [(x,y,pos_rad) for (x,y,_,_) in pos], pad=0, rad=pos_rad)
                neg_kernels[n], _ = norm_kernel_from_exemplars(results[n]["chan"], [(x,y,neg_rad) for (x,y,_,_) in neg], pad=0, rad=neg_rad)

            bayes_map=None
            for wi, n in zip(weights, names, strict=False):
                mp = multiscale_corr(results[n]["chan"], pos_kernels[n], scales=scales,
                                      use_antialias=use_antialias, use_simpson=use_simpson)
                mn = multiscale_corr(results[n]["chan"], neg_kernels[n], scales=scales,
                                      use_antialias=use_antialias, use_simpson=use_simpson)
                m  = wi * (mp - float(refine_beta) * mn)
                bayes_map = m if bayes_map is None else np.maximum(bayes_map, m)

            if gpu.USE_CUPY:
                fused_gpu = to_gpu(fused)
                bayes_gpu = to_gpu(bayes_map)
                f95 = float(cp.percentile(fused_gpu, 95.0))
                b95 = float(cp.percentile(bayes_gpu, 95.0))
            else:
                f95 = float(np.percentile(fused, 95.0))
                b95 = float(np.percentile(bayes_map, 95.0))

            scale = (f95+1e-6)/(b95+1e-6) if b95>0 else 1.0

            if gpu.USE_CUPY:
                bayes_norm = cp.clip(bayes_gpu * scale, 0, 1)
                bayes_norm = to_numpy(bayes_norm)
            else:
                bayes_norm = np.clip(bayes_map * scale, 0, 1)

            dets_bayes = peaks_from_map(bayes_norm, rad_use, thr_abs, nms_rad, mode=thr_mode, local_k=lk, local_ksize=lksz,
                                        topk_per_tile=int(thr_topk_tile), tile_factor=float(thr_tile_factor))
            used = [(x,y) for (x,y,_,_) in dets_raw]
            for (x,y,r,s) in dets_bayes:
                if all(math.hypot(x-ux, y-uy) > max(2.0, 0.5*rad_use) for (ux,uy) in used):
                    dets_bayes_added.append((x,y,r,s))
    t_bayes = time.perf_counter() - t1

    if dump is not None:
        try:
            ddir = dump.get("dir",".")
            os.makedirs(ddir, exist_ok=True)
            if dump.get("save_corr",0):
                vis = cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(ddir, "F0_corr_fused.png"), vis)
                best_name = names[0]
                best_corr = corr_maps[best_name]
                visb = cv2.normalize(best_corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(ddir, "F0_corr_best.png"), visb)
        except Exception:
            pass

    t_total = time.perf_counter() - t0_all
    if current_frame_idx < 10:
        print(f"[DET_PROF] chans:{t_chans*1000:.0f} hough:{t_hough*1000:.0f} log:{t_log*1000:.0f} score:{t_score*1000:.0f} corr:{t_corr*1000:.0f} fuse:{t_fuse*1000:.0f} peaks:{t_peaks*1000:.0f} bayes:{t_bayes*1000:.0f} total:{t_total*1000:.0f}")

    dbg = dict(rad=rad_use, nms=nms_rad, weights={n:float(w) for n,w in zip(names,weights, strict=False)},
               raw=len(dets_raw), bayes_added=len(dets_bayes_added),
               thr_mode=str(thr_mode), thr_abs=float(thr_abs), thr_local_k=float(lk),
               thr_local_ksize=int(lksz) if lksz is not None else 0, thr_topk_tile=int(thr_topk_tile),
               is_log_fallback=is_log_fallback)
    return dets_raw, dets_bayes_added, gray, chans, dbg

