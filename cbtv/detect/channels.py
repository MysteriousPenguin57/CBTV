"""Pruned from the research module ``detection_benchmark.py``: only the code reachable from the shipped
detectors (csv, ws_multi_vote / ws_multi_vote_gpu, ws_vgl_log, cellpose) is kept, unchanged.

Original module docstring:

Detection Benchmark Harness — Standalone detection-only testing for CBTV.

Tests 8 detection strategies across multiple videos, evaluates against ground truth,
and outputs comparison CSVs + optional visual overlays.

Usage:
    python detection_benchmark.py --video K01 --strategy all --frames 0,50,100
    python detection_benchmark.py --video all --strategy A,C,E --frames 0-400 --overlays
    python detection_benchmark.py --video K05,L01 --strategy all --frames 0-400

Strategies:
    A — Baseline (Hough → LoG → NCC kernel, current pipeline)
    B — CLAHE-enhanced Hough (CLAHE preprocessing before Hough)
    C — Consensus Mask + Distance Transform + Watershed (topological, no Hough)
    D — Adaptive Threshold + Contour Centroids (simple, no edge assumptions)
    E — CLAHE + Consensus Mask + DT/Watershed (fusion of B+C)
    F — Hough candidates filtered by DT validation (refine Hough via DT)
    G — LoG-only (no Hough at all, LoG as sole exemplar generator)
    H — NCC correlation + Watershed post-refinement (watershed on NCC peaks)

Does NOT touch hybrid_tracker.py. All detection logic is self-contained here.
"""

import cv2
import numpy as np

USE_WATERSHED_PEAKS = True    # fix 1 (ON by default; ws2k_* carved out)

USE_BORDER_PAD      = False   # fix 2 (OFF by default — no-op in practice)

USE_MEDIAN_KERNEL   = True    # fix 3 (ON by default)

USE_OBSTACLE_MASK   = False   # fix 4 (OFF by default — mask needs tightening)

# Phase 2E background-subtract test. When True, subtract_background(bgr) is
# called once per frame BEFORE build_channels(). Default OFF so the switch is
# ablation-gated. A separate CLI flag (--subtract-background) in the benchmark
# runner flips this. Mode below BACKGROUND_MODE_SKIP_BELOW per channel => that
# channel is already near-black and the subtraction is a no-op for it.
SUBTRACT_BACKGROUND = False

BACKGROUND_MODE_SKIP_BELOW = 5

# Phase 2F fix 2F-1: top-hat preprocessing before Otsu in classic_watershed.
# Applies to the watershed-family strategies. When True, classic_watershed
# replaces its input with cv2.morphologyEx(MORPH_TOPHAT, ellipse(2*rmax+1))
# before thresholding — suppresses structured background (funnel walls,
# bg-subtract halos) while keeping small bright features.
TOPHAT_THRESHOLD = False

# Phase 2F fix 2F-2: union Hough-circle exemplars from gray_eq AND V. When
# True, _hough_log_exemplars runs cv2.HoughCircles on both channels and
# dedups the combined list. Helps NCC pipelines on videos where gray-only
# Hough misses particle centers (K06 under bgon, L01 jitter).
HOUGH_MULTI_CHANNEL = False

# Phase 2F fix 2F-3: fixed-threshold bg-subtract. Instead of per-frame mode
# estimation, subtract FIXED_BG_MODE from every channel where the per-frame
# mode is > BACKGROUND_MODE_SKIP_BELOW (so PRIMARY still skips). Avoids
# jitter-induced drift in the per-frame mode estimate. Based on Stage 1
# data showing K01..L03 modes cluster tightly at 37-38 on every frame.
USE_FIXED_BG = False

FIXED_BG_MODE = 38

CLIP_NOISE_FLOOR = False

NOISE_FLOOR_LSB = 5

# Phase 2H fix 2H-1: frame-conditional clip gating. Phase 2G's unconditional
# clip rescued K06 (sparse foreground, ~5%) but regressed K04 (~25%) and K05
# (~30%) by amputating real signal in the 1-5 LSB anti-aliasing band on
# dense-foreground frames. With conditional gating the clip fires only when
# the channel's nonzero-pixel fraction is below the threshold. PRIMARY
# (mode 0) never fires; K06 sparse channels do; K04/K05 dense channels skip.
CLIP_FOREGROUND_THRESHOLD = 0.10

# Diagnostic: per-frame map of which channels fired the clip. Cleared by the
# runner each frame; populated by build_channels. Schema:
#     {'Gray': bool, 'V': bool, 'L': bool, 'B': bool}
CLIP_FIRED_CHANNELS: dict = {}

# ---------------------------------------------------------------------------
# Phase 2J flags (mechanism-targeted colour-jitter resilience)
# ---------------------------------------------------------------------------
# Continuous voting in _merge_votes -- when ON, watershed-multi-vote keeps
# clusters whose source-coverage is >= 1/n_sources and weights the merged
# score by the consensus fraction. When OFF, behaves as the original 2-of-3
# binary gate. Affects ws_multi_vote and ws_multi_circ.
CONTINUOUS_VOTING = False

# Gaussian-weighted centroid refinement -- when ON, watershed-family
# strategies post-process their detection list with gaussian_centroid_refine
# (intensity-weighted centroid in a window of radius rmax around each
# detection). NCC-family strategies pass through unchanged.
GAUSSIAN_CENTROID = False

# Per-call flag: the _temporal strategy wrappers set this to True before
# calling through so that build_kernel / _rank_channels knows to apply the
# EMA. Normal (non-_temporal) strategies leave it False. This lets the
# module flag TEMPORAL_KERNEL act as the global gate while individual
# _temporal strategies consume it.
TEMPORAL_KERNEL_ACTIVE = False

def subtract_background(bgr, skip_below=None, mode_override=None):
    """Phase 2E: per-channel background subtraction using histogram mode.

    For each of the 3 BGR channels independently:
      1. Find the mode (most-common pixel value) -- this is the per-channel
         background estimate.
      2. If mode is below ``skip_below`` (default BACKGROUND_MODE_SKIP_BELOW,
         which is 5), the channel is already near-black; skip it.
      3. Otherwise subtract the mode (or ``mode_override`` when given) from
         every pixel in that channel and clip to [0, 255].

    Phase 2F fix 2F-3 (``USE_FIXED_BG``): when the module flag USE_FIXED_BG
    is True AND the caller has not provided its own mode_override, substitute
    the per-channel mode with ``FIXED_BG_MODE`` (default 38). The skip-below
    check still runs on the per-frame estimate, so PRIMARY (mode ≈ 0) still
    passes through unchanged. Motivation: on jittered videos (L01, L02) the
    per-frame mode drifts ±1 LSB frame-to-frame because the noisy grey floor
    shifts the histogram peak; a fixed 38 avoids that residual drift.

    Why mode, not min or median
      min: too noisy (a single dark pixel sets it to 0).
      median: wrong when particles occupy >50% of frame (shifts toward
              particle intensities).
      mode: robust as long as background dominates pixel count, which holds
            on every CBTV video in the registry (viewport < 50% of frame).

    Called from the benchmark runner AFTER ``cap.read()`` and BEFORE
    ``build_channels(bgr)``. Intentionally NOT invoked inside build_channels
    so the ablation gate is a single file-level call rather than scattered
    inside channel prep.
    """
    if bgr is None:
        return bgr
    if skip_below is None:
        skip_below = BACKGROUND_MODE_SKIP_BELOW
    out = bgr.copy()
    for c in range(out.shape[2]):
        ch = out[:, :, c]
        counts = np.bincount(ch.ravel(), minlength=256)
        mode = int(np.argmax(counts))
        if mode < int(skip_below):
            continue
        # Phase 2F fix 2F-3: replace per-frame mode with a fixed value.
        if mode_override is not None:
            mode_used = int(mode_override)
        elif USE_FIXED_BG:
            mode_used = int(FIXED_BG_MODE)
        else:
            mode_used = mode
        # Subtract and clip. Use int16 to avoid uint8 wraparound.
        sub = ch.astype(np.int16) - mode_used
        np.clip(sub, 0, 255, out=sub)
        out[:, :, c] = sub.astype(np.uint8)
    return out

def build_channels(bgr):
    """Build multi-channel representation (mirrors hybrid_tracker.build_channels).

    Phase 2H fix 2H-1: ``CLIP_NOISE_FLOOR`` now fires *conditionally per
    channel* — only when the channel's nonzero-pixel fraction (count of
    pixels strictly above NOISE_FLOOR_LSB, divided by total pixel count) is
    below ``CLIP_FOREGROUND_THRESHOLD``. The "channel is sparse so the
    histogram is dominated by compression-noise" condition that the clip
    cures only holds on sparse channels; on dense-foreground channels
    (K04/K05) the same clip amputates the 1-5 LSB anti-aliasing band that
    contains real edge signal. Per-channel gating lets a single video have
    K06's sparse V channel fire while K04's dense Gray channel skips.

    Diagnostic: which channels fired is recorded in
    ``CLIP_FIRED_CHANNELS`` (module global) for the runner to log. Reset
    each frame.
    """
    global CLIP_FIRED_CHANNELS
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    labL = lab[:, :, 0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsvS = hsv[:, :, 1]
    hsvV = hsv[:, :, 2]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    yccY = ycrcb[:, :, 0]

    # Phase 2H fix 2H-1: per-channel conditional clip.
    fired = {'Gray': False, 'V': False, 'L': False, 'B': False}
    if CLIP_NOISE_FLOOR:
        thr = int(NOISE_FLOOR_LSB)
        frac_thr = float(CLIP_FOREGROUND_THRESHOLD)

        def _maybe_clip(ch_arr, name):
            # frac_above = pixels strictly above the noise band / total pixels.
            # Equivalent to "fraction of foreground pixels" under the noise
            # floor's definition. Cheap with a single boolean reduce.
            frac_above = float((ch_arr > thr).sum()) / float(ch_arr.size)
            if frac_above < frac_thr:
                out = ch_arr.copy()
                out[out <= thr] = 0
                fired[name] = True
                return out
            # Dense foreground -> preserve sub-threshold pixels.
            return ch_arr

        gray = _maybe_clip(gray, 'Gray')
        labL = _maybe_clip(labL, 'L')
        hsvV = _maybe_clip(hsvV, 'V')
        yccY = _maybe_clip(yccY, 'B')

    # Make the per-frame fire decisions visible to the runner / report.
    CLIP_FIRED_CHANNELS = fired

    # Histogram equalization — plain equalizeHist, NOT CLAHE
    # (matches hybrid_tracker.py equalize() at line 671)
    labL = cv2.equalizeHist(labL)
    gray_eq = cv2.equalizeHist(gray)
    hsvV = cv2.equalizeHist(hsvV)
    yccY = cv2.equalizeHist(yccY)

    channels = {'L': labL, 'S': hsvS, 'V': hsvV, 'B': yccY, 'Gray': gray_eq}
    return channels, gray_eq
