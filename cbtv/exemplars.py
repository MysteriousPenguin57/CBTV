"""Exemplar generation (Hough/multiframe/fallback) + kernel-from-exemplars."""
import cv2
import numpy as np

from . import gpu
from .correlation import dedup_circles, ncc
from .gpu import cp, to_gpu, to_numpy


def hough_exemplars_dual(gray, rmin=6, rmax=10, param2=12, minDist=8, nmax=400, polarity="auto"):
    """Run HoughCircles on both inverted and normal grayscale images and deduplicate results.

    Pipeline context:
        Called by: multiframe_hough_exemplars, user_detection_with_bayes_adapt
        Stage: detection

    Hough detections are used as *exemplars* — candidate particle centre
    locations — not as final detections.  The returned (x, y, r) tuples are
    later passed to norm_kernel_from_exemplars to build NCC correlation kernels.
    Running on both the inverted and normal image compensates for particles
    that appear as bright blobs on dark backgrounds or vice versa.

    Args:
        gray:     Grayscale frame [np.ndarray, uint8, HxW]
        rmin:     Minimum circle radius in pixels [int, default 6]
        rmax:     Maximum circle radius in pixels [int, default 10]
        param2:   HoughCircles accumulator threshold (lower = more detections)
                  [int, default 12]
        minDist:  Minimum centre-to-centre distance between circles [int, default 8]
        nmax:     Maximum number of exemplars to return [int, default 400]
        polarity: Which image polarity to run Hough on:
                  "inv"  — inverted image only,
                  "norm" — normal image only,
                  "auto" / "both" — both images (default "auto") [str]

    Returns:
        Deduplicated list of (x, y, r) exemplar tuples [list[tuple[float, float, float]]]
    """
    ex_all=[]
    if polarity in ("inv","auto","both"):
        inv = cv2.subtract(255, gray)
        cir = cv2.HoughCircles(inv, cv2.HOUGH_GRADIENT, dp=1.0, minDist=minDist,
                               param1=100, param2=param2, minRadius=rmin, maxRadius=rmax)
        if cir is not None:
            ex_all += [(float(x),float(y),float(r)) for x,y,r in np.round(cir[0]).astype(int)[:nmax]]
    if polarity in ("norm","auto","both"):
        cir = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=minDist,
                               param1=100, param2=param2, minRadius=rmin, maxRadius=rmax)
        if cir is not None:
            ex_all += [(float(x),float(y),float(r)) for x,y,r in np.round(cir[0]).astype(int)[:nmax]]
    if not ex_all:
        return []
    ex_all = dedup_circles(ex_all, dist_thresh=3.0)
    return ex_all[:nmax]

def multiframe_hough_exemplars(frames_gray, frame_idx, rmin=6, rmax=10, param2=12, minDist=8, nmax=800, polarity="auto", offset=5):
    """
    Gather Hough circle candidates from multiple frames (t-offset, t, t+offset)
    and deduplicate to get a richer candidate pool

    Args:
        frames_gray: Rolling buffer of gray frames (most recent at end)
        frame_idx: Current frame index (absolute frame number)
        offset: Frame offset (default 5 for t-5, t, t+5)

    Returns:
        Deduplicated list of exemplars from multiple frames
    """
    ex_all = []
    num_frames = len(frames_gray)

    if num_frames == 0:
        return []

    # For rolling buffer: most recent frame is at index -1
    # We want frames at relative positions: -offset, 0, +offset from current
    # In buffer terms: current is at -1, so we need indices relative to that

    # Buffer positions to check (relative to buffer end)
    buffer_offsets = []
    current_buffer_idx = num_frames - 1  # Current frame at end

    for off in [-offset, 0, offset]:
        buf_idx = current_buffer_idx + off
        # Clamp to available buffer
        buf_idx = max(0, min(num_frames - 1, buf_idx))
        if buf_idx not in buffer_offsets:  # Avoid duplicates at edges
            buffer_offsets.append(buf_idx)

    for idx in buffer_offsets:
        gray = frames_gray[idx]
        ex = hough_exemplars_dual(gray, rmin=rmin, rmax=rmax, param2=param2,
                                   minDist=minDist, nmax=nmax//3, polarity=polarity)
        ex_all.extend(ex)

    if not ex_all:
        return []

    # Deduplicate across all frames
    ex_all = dedup_circles(ex_all, dist_thresh=3.0)
    return ex_all[:nmax]

def propose_exemplars_fallback(gray, r_guess=10, nmax=300, win=9):
    """Generate exemplars as a last resort via Laplacian peak finding.

    Pipeline context:
        Called by: user_detection_with_bayes_adapt (when Hough and LoG both
                   return too few exemplars)
        Stage: detection

    Applies a small Gaussian blur, computes the absolute Laplacian response,
    and extracts local maxima using morphological dilation NMS.  Peaks are
    collected in descending response order; a circular exclusion mask prevents
    the same spatial region being selected twice.  All returned exemplars share
    the same fixed radius r_guess.

    Args:
        gray:    Grayscale frame [np.ndarray, uint8, HxW]
        r_guess: Radius assigned to every returned exemplar [float, default 10]
        nmax:    Maximum number of exemplars to return [int, default 300]
        win:     NMS window half-size in pixels; actual window is (win|1) x (win|1)
                 [int, default 9]

    Returns:
        List of (x, y, r_guess) exemplar tuples sorted by descending Laplacian
        response [list[tuple[float, float, float]]]
    """
    blur = cv2.GaussianBlur(gray, (0,0), 1.2)
    lap  = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)

    if gpu.USE_CUPY:
        lap_gpu = to_gpu(lap)
        lap = to_numpy(cp.abs(lap_gpu))
    else:
        lap = np.abs(lap)

    k = max(3, int(win)|1)
    mx = cv2.dilate(lap, cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    ys, xs = np.where(lap>=mx-1e-12)
    vals = lap[ys, xs]
    if vals.size==0: return []
    order = np.argsort(-vals, kind="stable")
    H,W = gray.shape
    taken = np.zeros((H,W), np.uint8)
    out=[]
    for idx in order:
        x,y = int(xs[idx]), int(ys[idx])
        if taken[y,x]: continue
        out.append((float(x), float(y), float(r_guess)))
        cv2.circle(taken, (x,y), int(max(4, r_guess)), 1, -1)
        if len(out)>=nmax: break
    return out

def intelligent_nms_radius(exemplars, base_factor=0.6):
    """Intelligent NMS radius based on exemplar statistics"""
    if not exemplars:
        return 5

    radii = np.array([r for _,_,r in exemplars])
    median_r = float(np.median(radii))
    std_r = float(np.std(radii))

    cv = std_r / (median_r + 1e-6)
    adjustment = 0.2 * min(1.0, cv / 0.3)
    factor = base_factor + adjustment

    nms_rad = max(4, int(factor * median_r))
    return nms_rad

def score_exemplar_quality_gpu(chan, exemplars, rad):
    """Score multiple exemplars in batch using GPU acceleration"""
    if not exemplars:
        return []

    H, W = chan.shape[:2]
    r_int = int(round(rad))
    n = len(exemplars)

    if gpu.USE_CUPY and n > 0:
        try:
            # Transfer channel to GPU once
            chan_gpu = cp.asarray(chan.astype(np.float32))

            # Compute gradients on GPU
            gx = cp.diff(chan_gpu, axis=1, prepend=chan_gpu[:, :1])
            gy = cp.diff(chan_gpu, axis=0, prepend=chan_gpu[:1, :])
            grad_mag = cp.sqrt(gx**2 + gy**2)

            # Vectorized center coordinates
            cx = cp.asarray([round(x) for x, y, r in exemplars], dtype=cp.int32)
            cy = cp.asarray([round(y) for x, y, r in exemplars], dtype=cp.int32)

            # Filter out elements whose bounding boxes extend out of the image by a large margin
            # However, for advanced indexing clip is safer
            idx_x = cp.arange(-r_int, r_int)
            idx_y = cp.arange(-r_int, r_int)

            grid_x = cx[:, None, None] + idx_x[None, None, :]
            grid_y = cy[:, None, None] + idx_y[None, :, None]

            # True out-of-bounds tracking to match CPU loop boundary checks
            invalid_mask = (grid_x[..., -1] - grid_x[..., 0] < 4) | (grid_y[..., -1] - grid_y[..., 0] < 4)
            invalid_mask = invalid_mask.get()

            # Clip to image boundaries
            grid_x = cp.clip(grid_x, 0, W-1)
            grid_y = cp.clip(grid_y, 0, H-1)

            # Extract ALL patches in one massive kernel parallel read
            patches = chan_gpu[grid_y, grid_x]
            g_patches = grad_mag[grid_y, grid_x]

            # Batch Compute Variance across spatial dimensions -> axis=(1,2)
            variances = cp.var(patches, axis=(1, 2))

            # Batch Compute Mean Gradient
            g_scores = cp.mean(g_patches, axis=(1, 2))

            # Combine
            batch_scores = (variances * 0.6) + (g_scores * 0.4)
            scores = cp.asnumpy(batch_scores).tolist()

            # Zero out any effectively invalid ones
            for i in range(n):
                if invalid_mask[i,0] if len(invalid_mask.shape)>1 else invalid_mask[i]:
                    scores[i] = 0.0

            return scores

        except Exception:
            # Fall back to CPU
            pass

    # CPU fallback
    scores = []
    for x, y, r in exemplars:
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r_int), min(W, xi + r_int)
        y0, y1 = max(0, yi - r_int), min(H, yi + r_int)

        if x1 - x0 < 4 or y1 - y0 < 4:
            scores.append(0.0)
            continue

        patch = chan[y0:y1, x0:x1].astype(np.float32)
        variance = float(np.var(patch))

        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag_cpu = np.sqrt(gx**2 + gy**2)
        gradient_score = float(np.mean(grad_mag_cpu))

        score = variance * 0.6 + gradient_score * 0.4
        scores.append(score)

    return scores

def select_best_exemplars(chan, exemplars, rad, nmax=120, min_quality_score=None):
    """
    Select best exemplars based on quality scoring

    Args:
        chan: Channel image
        exemplars: List of (x, y, r) tuples
        rad: Target radius
        nmax: Maximum number of exemplars to select
        min_quality_score: Minimum quality threshold (None = auto-select top nmax)
    """
    if len(exemplars) == 0:
        return exemplars

    # Score all exemplars in batch
    scores = score_exemplar_quality_gpu(chan, exemplars, rad)

    # Zip scores with exemplars and sort by quality
    scored = list(zip(scores, exemplars, strict=False))
    scored.sort(reverse=True, key=lambda x: x[0])

    # Apply minimum quality threshold if specified (more selective for multi-frame)
    if min_quality_score is not None:
        scored = [(s, ex) for s, ex in scored if s >= min_quality_score]
        if len(scored) == 0:
            # If too strict, fall back to top exemplars only
            scored = list(zip(scores, exemplars, strict=False))
            scored.sort(reverse=True, key=lambda x: x[0])

    # Return top nmax after quality filtering
    return [ex for _, ex in scored[:nmax]]

def norm_kernel_from_exemplars(chan, exemplars, pad=0, rad=None, nmax=120):
    """Build a normalized NCC correlation kernel by averaging quality-selected exemplar patches.

    Pipeline context:
        Called by: user_detection_with_bayes_adapt
        Stage: correlation

    Selects the best exemplars via select_best_exemplars, extracts a patch
    around each one at its actual radius, resizes every patch to a common
    target size derived from the median radius (scale-invariant), zero-mean
    and unit-variance normalises each patch, then averages them to produce a
    single representative kernel.  The averaged kernel is finally L2-normalised.

    Args:
        chan:      Single-channel detection image to crop patches from
                  [np.ndarray, float32, HxW]
        exemplars: List of (x, y, r) candidate locations
                  [list[tuple[float, float, float]]]
        pad:       Extra padding in pixels added to each side of the patch
                  beyond the particle radius [int, default 0]
        rad:       Override target radius; if None, median of exemplar radii
                  is used [int | None, default None]
        nmax:      Maximum number of exemplars to use when building the kernel
                  [int, default 120]

    Returns:
        Tuple of (kernel, radius) where kernel is the averaged L2-normalised
        template and radius is the target radius used for sizing
        [tuple[np.ndarray, int]]
    """
    if not exemplars: raise RuntimeError("No exemplars found")

    if rad is None:
        radii = [r for _,_,r in exemplars]
        if gpu.USE_CUPY:
            rad = int(cp.median(cp.array(radii)))
        else:
            rad = int(np.median(radii))

    # Select best exemplars based on quality
    best_exemplars = select_best_exemplars(chan, exemplars, rad, nmax)

    # Refinement is omitted here to save ~800ms. Edge fit is already applied at the very end of the detection pipeline.

    size = 2*(rad+pad)
    patches=[]
    H,W = chan.shape[:2]

    for (x,y,r) in best_exemplars:
        xi, yi = int(round(x)), int(round(y))

        # SCALE-INVARIANT KERNEL: Extract patch at actual radius, then scale to median radius
        # This ensures all exemplars contribute equally regardless of their size
        actual_size = 2*(int(r)+pad)
        x0, x1 = max(0, xi-(int(r)+pad)), min(W, xi+(int(r)+pad))
        y0, y1 = max(0, yi-(int(r)+pad)), min(H, yi+(int(r)+pad))
        patch = chan[y0:y1, x0:x1].astype(np.float32)

        # Place in canvas sized for actual radius
        temp_canvas = np.zeros((actual_size, actual_size), np.float32)
        cy0, cx0 = (actual_size-(y1-y0))//2, (actual_size-(x1-x0))//2
        temp_canvas[cy0:cy0+(y1-y0), cx0:cx0+(x1-x0)] = patch

        # Scale to target size using interpolation (for scale invariance)
        if actual_size != size:
            canvas = cv2.resize(temp_canvas, (size, size), interpolation=cv2.INTER_LINEAR)
        else:
            canvas = temp_canvas

        p = (canvas - canvas.mean())/(canvas.std()+1e-6)
        patches.append(p)

    k = np.mean(np.stack(patches,0),0)
    k -= k.mean()
    k /= (np.sqrt((k**2).sum())+1e-6)

    return k.astype(np.float32), rad

def score_channel(chan32, exemplars, kernel32, rad, holdout=0.3, rng=None):
    """Score a detection channel by measuring how well the kernel localises held-out exemplars.

    Pipeline context:
        Called by: user_detection_with_bayes_adapt (channel selection loop)
        Stage: detection

    Runs a full NCC correlation of kernel32 over chan32, then samples the
    correlation response at each held-out exemplar location.  The mean
    response is returned as the channel quality score; higher scores indicate
    that the channel produces sharper NCC peaks at known particle locations.

    Args:
        chan32:    Detection channel image [np.ndarray, float32, HxW]
        exemplars: Full list of (x, y, r) exemplar locations; a random
                  holdout fraction is used for scoring [list[tuple]]
        kernel32:  Normalised NCC kernel built from the training exemplars
                  [np.ndarray, float32]
        rad:       Kernel radius used to convert peak centre to correlation
                  map lookup coordinate [int]
        holdout:   Fraction (0–1) of exemplars reserved as test set; at least
                  5 exemplars are always used [float, default 0.3]

    Returns:
        Mean NCC response at held-out exemplar positions; 0.0 if no exemplars
        [float]
    """
    n = len(exemplars)
    if n<=0: return 0.0
    m = max(5, int(n*holdout)) if n>5 else n

    idx = np.arange(n)
    # rng: np.random.Generator from --seed; None = legacy global state.
    (rng if rng is not None else np.random).shuffle(idx)

    val = [exemplars[i] for i in idx[:m]]
    corr = ncc(chan32, kernel32)
    scores=[]
    for (x,y,_) in val:
        xi = int(max(0,min(corr.shape[1]-1, x-rad)))
        yi = int(max(0,min(corr.shape[0]-1, y-rad)))
        scores.append(float(corr[yi, xi]))

    return float(np.mean(scores)) if scores else 0.0

