"""NCC correlation engine: channels, kernels, peaks, subpixel."""
import math

import cv2
import numpy as np

from . import gpu
from .gpu import cp, to_gpu, to_numpy


def equalize(img):
    """Apply histogram equalization to a grayscale image.

    Pipeline context:
        Called by: build_channels
        Stage: correlation

    Args:
        img: Grayscale or colour image array [np.ndarray, uint8]

    Returns:
        Histogram-equalized image if input is 2-D (grayscale), otherwise
        the input is returned unchanged [np.ndarray]
    """
    return cv2.equalizeHist(img) if img.ndim==2 else img

def build_channels(bgr):
    """Convert a BGR frame into five histogram-equalized detection channels.

    Pipeline context:
        Called by: user_detection_with_bayes_adapt, norm_kernel_from_exemplars,
                   score_channel
        Stage: correlation

    Produces channels:
        L    - CIE L* lightness (BGR→LAB channel 0), equalized
        S    - HSV saturation (channel 1), not equalized
        V    - HSV value / brightness (channel 2), equalized
        B    - YCrCb luma Y (channel 0), equalized
        Gray - standard grayscale, equalized

    Args:
        bgr: Full-colour video frame in BGR format [np.ndarray, uint8, HxWx3]

    Returns:
        Tuple of (channels_dict, gray) where channels_dict maps channel name
        to 2-D uint8 array and gray is the equalized grayscale image
        [tuple[dict[str, np.ndarray], np.ndarray]]
    """
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    labL  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:,:,0]
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); hsvS=hsv[:,:,1]; hsvV=hsv[:,:,2]
    yccY  = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
    labL = equalize(labL); gray = equalize(gray); hsvV = equalize(hsvV); yccY = equalize(yccY)
    return dict(L=labL, S=hsvS, V=hsvV, B=yccY, Gray=gray), gray

def ncc(img32, k32, use_simpson=False, use_gaussian=False, use_bandpass=False, use_phase=False, use_gradient=False):
    """
    NCC with OpenCL GPU acceleration and optional kernel preprocessing/alternative correlation.

    Args:
        use_simpson: Apply Simpson's 1/3 rule weighting (piecewise: corners=1, edges=2, interior=4)
        use_gaussian: Apply smooth Gaussian weighting (alternative to Simpson, smoother)
        use_bandpass: Apply bandpass filtering to kernel (remove low/high freq noise)
        use_phase: Use phase correlation (frequency domain, sharp peaks)
        use_gradient: Use gradient-based correlation (emphasize structure)
    """
    # Apply kernel preprocessing (mutually exclusive, priority order)
    if use_gaussian:
        k32 = apply_gaussian_weights(k32)
    elif use_simpson:
        k32 = apply_simpson_weights(k32)

    if use_bandpass:
        k32 = apply_bandpass_filter(k32)

    # Use alternative correlation method if specified
    if use_phase:
        return phase_correlation(img32.astype(np.float32), k32)
    elif use_gradient:
        return gradient_correlation(img32.astype(np.float32), k32)

    # Standard NCC
    if gpu.USE_OPENCL:
        img_gpu = cv2.UMat(img32.astype(np.float32))
        k_gpu = cv2.UMat(k32.astype(np.float32))
        result = cv2.matchTemplate(img_gpu, k_gpu, cv2.TM_CCORR_NORMED)
        return result.get()
    return cv2.matchTemplate(img32.astype(np.float32), k32.astype(np.float32), cv2.TM_CCORR_NORMED)

def multiscale_corr(img32, kernel32, scales=(0.9,1.0,1.1), use_antialias=False, use_simpson=False,
                   use_gaussian=False, use_bandpass=False, use_phase=False, use_gradient=False):
    """Multiscale correlation with GPU acceleration, optional anti-aliasing, and alternative methods"""
    def _pad_to_img(m, ks):
        """Pad correlation map to match original image dimensions."""
        if m.shape == img32.shape[:2]: return m
        h, w = ks.shape[:2]
        t = max(0, h // 2)
        b = max(0, img32.shape[0] - m.shape[0] - t)
        left = max(0, w // 2)
        r = max(0, img32.shape[1] - m.shape[1] - left)
        return cv2.copyMakeBorder(m, t, b, left, r, cv2.BORDER_CONSTANT, value=0.0)
    if gpu.USE_OPENCL:
        img_gpu = cv2.UMat(img32.astype(np.float32))
        maps = []
        for s in scales:
            if use_antialias and s < 1.0:
                ks = antialias_downsample(kernel32, s).astype(np.float32)
            else:
                ks = cv2.resize(kernel32, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR).astype(np.float32)
            ks_gpu = cv2.UMat(ks)
            m = cv2.matchTemplate(img_gpu, ks_gpu, cv2.TM_CCORR_NORMED).get()
            maps.append(_pad_to_img(m, ks))

        if gpu.USE_CUPY:
            corr_gpu = to_gpu(maps[0])
            for m in maps[1:]:
                corr_gpu = cp.maximum(corr_gpu, to_gpu(m))
            return to_numpy(corr_gpu)
        else:
            corr = maps[0].copy()
            for m in maps[1:]:
                np.maximum(corr, m, out=corr)
            return corr

    # CPU fallback
    maps = []
    for s in scales:
        if use_antialias and s < 1.0:
            ks = antialias_downsample(kernel32, s).astype(np.float32)
        else:
            ks = cv2.resize(kernel32, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        m = ncc(img32, ks, use_simpson=use_simpson, use_gaussian=use_gaussian,
                use_bandpass=use_bandpass, use_phase=use_phase, use_gradient=use_gradient)
        maps.append(_pad_to_img(m, ks))

    if gpu.USE_CUPY:
        corr_gpu = to_gpu(maps[0])
        for m in maps[1:]:
            corr_gpu = cp.maximum(corr_gpu, to_gpu(m))
        return to_numpy(corr_gpu)
    else:
        corr = maps[0].copy()
        for m in maps[1:]:
            np.maximum(corr, m, out=corr)
        return corr

def nms_mask(corr, nms=5):
    """Create a binary non-maximum suppression mask from a correlation map.

    Pipeline context:
        Called by: peaks_from_map, peaks_to_list (indirectly via peaks_from_map)
        Stage: correlation

    A pixel survives NMS if it equals the maximum within the surrounding
    (2*nms+1) x (2*nms+1) rectangular neighbourhood (implemented via
    morphological dilation with a 1e-12 tolerance).

    Args:
        corr: 2-D NCC correlation response map [np.ndarray, float32]
        nms:  Half-width of the suppression window in pixels; total window
              is (2*nms+1)^2 [int, default 5]

    Returns:
        Boolean mask where True marks local-maximum pixels [np.ndarray, bool]
    """
    k = 2*nms+1
    mx = cv2.dilate(corr, cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    return (corr>=mx-1e-12)

def peaks_to_list(corr, mask, r, enable_subpixel=False):
    """Convert an NMS mask and correlation map into a list of (x, y, r, score) detections.

    Pipeline context:
        Called by: peaks_from_map
        Stage: correlation

    Each True pixel in mask becomes a detection at integer (or sub-pixel if
    enabled) coordinates with the NCC score sampled from corr at that location.

    Args:
        corr:            2-D NCC correlation map [np.ndarray or CuPy array, float32]
        mask:            Boolean NMS mask of surviving pixels [np.ndarray or CuPy array, bool]
        r:               Radius to assign to every detection [float]
        enable_subpixel: If True, refine peak centres with 2-D parabolic
                         sub-pixel interpolation [bool, default False]

    Returns:
        List of (x, y, r, score) tuples, one per surviving peak
        [list[tuple[float, float, float, float]]]
    """
    corr = to_numpy(corr)
    mask = to_numpy(mask)
    ys, xs = np.where(mask)

    if not enable_subpixel:
        return [(float(x), float(y), float(r), float(corr[y,x])) for y,x in zip(ys,xs, strict=False)]

    # Sub-pixel refinement enabled
    detections = []
    for y, x in zip(ys, xs, strict=False):
        x_subpix, y_subpix = subpixel_refine_2d_parabolic(corr, y, x)
        detections.append((float(x_subpix), float(y_subpix), float(r), float(corr[y,x])))
    return detections

def dedup_circles(circles, dist_thresh=4.0):
    """Remove duplicate circle detections, keeping the first occurrence within the distance threshold.

    Pipeline context:
        Called by: hough_exemplars_dual, multiframe_hough_exemplars,
                   propose_exemplars_fallback (indirectly)
        Stage: filtering

    Iterates circles in their original order (assumed highest-score first).
    Any subsequent circle whose centre is within dist_thresh pixels of an
    already-accepted circle is discarded.

    Args:
        circles:     List of (x, y, r) circle tuples [list[tuple[float, float, float]]]
        dist_thresh: Maximum centre-to-centre distance in pixels for two
                     circles to be considered duplicates [float, default 4.0]

    Returns:
        Deduplicated list of (x, y, r) tuples [list[tuple[float, float, float]]]
    """
    out=[]
    taken=np.zeros(len(circles), np.uint8)
    for i,(x,y,r) in enumerate(circles):
        if taken[i]: continue
        out.append((x,y,r))
        for j in range(i+1, len(circles)):
            if taken[j]: continue
            x2,y2,r2 = circles[j]
            if math.hypot(x-x2,y-y2) <= dist_thresh:
                taken[j]=1
    return out

def subpixel_refine_2d_parabolic(corr, y_int, x_int):
    """
    Sub-pixel peak refinement using 2D parabolic fitting.

    Fits a 2D parabola to the 3x3 neighborhood around (x_int, y_int)
    and finds the sub-pixel maximum location.

    Theory: A correlation peak can be modeled as a 2D parabola near its maximum.
    f(x,y) = a*x^2 + b*y^2 + c*x + d*y + e*x*y + f

    The maximum of this parabola gives the sub-pixel peak location with
    typical accuracy of 0.05-0.1 pixels (vs 1.0 pixel for integer peaks).

    Returns: (x_subpixel, y_subpixel) or (x_int, y_int) if refinement fails
    """
    h, w = corr.shape

    # Need at least 1 pixel border for 3x3 neighborhood
    if x_int < 1 or x_int >= w-1 or y_int < 1 or y_int >= h-1:
        return float(x_int), float(y_int)

    # Extract 3x3 neighborhood
    neighborhood = corr[y_int-1:y_int+2, x_int-1:x_int+2]

    # Check if we have a valid peak (center should be maximum or near-maximum)
    center_val = neighborhood[1, 1]
    max_val = neighborhood.max()
    if center_val < max_val * 0.95:  # Center not dominant enough
        return float(x_int), float(y_int)

    # 2D parabolic fit using finite differences
    # Second derivatives (curvature)
    fxx = neighborhood[1, 2] - 2*neighborhood[1, 1] + neighborhood[1, 0]
    fyy = neighborhood[2, 1] - 2*neighborhood[1, 1] + neighborhood[0, 1]

    # First derivatives (slope)
    fx = (neighborhood[1, 2] - neighborhood[1, 0]) / 2.0
    fy = (neighborhood[2, 1] - neighborhood[0, 1]) / 2.0

    # Cross derivative
    fxy = (neighborhood[2, 2] - neighborhood[2, 0] - neighborhood[0, 2] + neighborhood[0, 0]) / 4.0

    # Solve for sub-pixel offset using Newton's method
    # [fxx fxy] [dx]   [fx]
    # [fxy fyy] [dy] = [fy]

    det = fxx * fyy - fxy * fxy

    # Check for numerical stability
    if abs(det) < 1e-10:
        return float(x_int), float(y_int)

    # Sub-pixel offsets
    dx = -(fyy * fx - fxy * fy) / det
    dy = -(fxx * fy - fxy * fx) / det

    # Constrain offsets to reasonable range (±0.5 pixels)
    dx = np.clip(dx, -0.5, 0.5)
    dy = np.clip(dy, -0.5, 0.5)

    return float(x_int + dx), float(y_int + dy)

def antialias_downsample(img, scale):
    """
    Anti-aliased image downsampling using Gaussian pre-filtering.

    Theory: Nyquist-Shannon sampling theorem states that downsampling
    without proper low-pass filtering causes aliasing (high frequencies
    fold back into low frequency range, creating artifacts).

    This function:
    1. Applies Gaussian blur to remove frequencies above new Nyquist limit
    2. Downsamples using LINEAR interpolation

    The sigma of the Gaussian is chosen based on the downsampling factor:
    sigma = sqrt((1/scale)^2 - 1) / 2

    This ensures smooth, artifact-free downsampling.

    Args:
        img: Input image (numpy array)
        scale: Downsampling factor (< 1.0 for downsample, >= 1.0 returns img)

    Returns:
        Downsampled image (or original if scale >= 1.0)
    """
    if scale >= 1.0:
        # No downsampling needed, just resize normally
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Calculate optimal Gaussian sigma for this downsampling factor
    # Based on Nyquist theory: filter out frequencies above new Nyquist limit
    sigma = math.sqrt(max(0.0, (1.0/scale)**2 - 1.0)) / 2.0

    # Apply Gaussian blur (low-pass filter)
    if sigma > 0.5:
        # Use at least 3-sigma kernel width for good frequency response
        ksize = int(math.ceil(sigma * 3) * 2 + 1)
        ksize = max(3, ksize | 1)  # Ensure odd and at least 3
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    else:
        blurred = img

    # Now downsample the pre-filtered image
    return cv2.resize(blurred, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def apply_simpson_weights(kernel):
    """
    Apply 2D Simpson's 1/3 rule weights to improve numerical integration accuracy.

    Theory: Simpson's rule provides better quadrature for smooth functions by
    weighting interior points more heavily. For 2D integration:
    - Corner points: weight 1
    - Edge points (not corners): weight 2
    - Interior points: weight 4

    This may improve correlation accuracy for smooth particle profiles.
    """
    h, w = kernel.shape
    if h < 3 or w < 3:
        return kernel  # Too small for Simpson's rule

    # Create weight matrix
    weights = np.ones_like(kernel, dtype=np.float32)

    # Interior points get weight 4
    if h > 2 and w > 2:
        weights[1:-1, 1:-1] = 4.0

    # Edge points (not corners) get weight 2
    if h > 2:
        weights[1:-1, 0] = 2.0
        weights[1:-1, -1] = 2.0
    if w > 2:
        weights[0, 1:-1] = 2.0
        weights[-1, 1:-1] = 2.0

    # Corners already have weight 1

    # Apply weights and renormalize
    weighted = kernel * weights
    return weighted / (weights.sum() / kernel.size)  # Preserve normalization

def apply_gaussian_weights(kernel):
    """
    Apply Gaussian weighting to kernel - smoother alternative to Simpson's rule.

    Emphasizes center region with smooth Gaussian falloff rather than piecewise weights.
    This avoids sharp discontinuities that can affect correlation peak magnitudes.

    Benefits:
    - Smooth weight transition (vs Simpson's sharp 1/2/4 steps)
    - Less disruption to correlation thresholds
    - Still emphasizes center region for better localization
    """
    h, w = kernel.shape
    if h < 3 or w < 3:
        return kernel

    # Create Gaussian weight matrix centered on kernel
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    sigma = min(h, w) / 4.0  # Gaussian width ~quarter of kernel size

    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    gaussian = gaussian.astype(np.float32)

    # Apply Gaussian weights
    weighted = kernel * gaussian

    # Normalize to preserve total kernel mass
    return weighted / weighted.sum() * kernel.sum()

def apply_bandpass_filter(kernel, low_sigma=0.5, high_sigma=2.5):
    """
    Apply bandpass filter to kernel to remove DC component and high-frequency noise.

    This can improve correlation quality by isolating the relevant spatial frequencies
    that represent particle structure while rejecting:
    - Low frequencies (illumination gradients, background)
    - High frequencies (sensor noise, compression artifacts)

    Args:
        low_sigma: Gaussian sigma for low-pass (removes high freq noise)
        high_sigma: Gaussian sigma for high-pass (removes low freq trends)

    Returns:
        Bandpass-filtered kernel
    """
    h, w = kernel.shape
    if h < 5 or w < 5:
        return kernel  # Too small for meaningful filtering

    # Low-pass filter (removes high-freq noise)
    low_pass = cv2.GaussianBlur(kernel, (0, 0), low_sigma)

    # High-pass filter (removes low-freq trends)
    high_pass_base = cv2.GaussianBlur(kernel, (0, 0), high_sigma)

    # Bandpass = low_pass - high_pass_base
    bandpass = low_pass - high_pass_base

    # Normalize
    if np.abs(bandpass).max() > 1e-6:
        bandpass = bandpass / np.linalg.norm(bandpass) * np.linalg.norm(kernel)

    return bandpass.astype(np.float32)

def phase_correlation(img, kernel):
    """
    Phase correlation in frequency domain - alternative to spatial NCC.

    Computes normalized cross-power spectrum which gives sharp correlation peaks.
    More robust to illumination changes than spatial correlation.

    Args:
        img: Image (float32)
        kernel: Template kernel (float32)

    Returns:
        Correlation map (same size as would be returned by matchTemplate)
    """
    # Pad kernel to image size
    h_img, w_img = img.shape
    h_k, w_k = kernel.shape

    # Create padded kernel centered
    kernel_padded = np.zeros_like(img)
    y_offset = (h_img - h_k) // 2
    x_offset = (w_img - w_k) // 2
    kernel_padded[y_offset:y_offset+h_k, x_offset:x_offset+w_k] = kernel

    # Compute FFT
    F_img = np.fft.fft2(img)
    F_kernel = np.fft.fft2(kernel_padded)

    # Cross-power spectrum
    cross_power = F_img * np.conj(F_kernel)
    cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)

    # Inverse FFT
    correlation = np.fft.ifft2(cross_power_norm)
    correlation = np.fft.fftshift(np.abs(correlation))

    # Crop to match matchTemplate output size
    out_h, out_w = h_img - h_k + 1, w_img - w_k + 1
    y_start = h_img // 2 - out_h // 2
    x_start = w_img // 2 - out_w // 2
    correlation_cropped = correlation[y_start:y_start+out_h, x_start:x_start+out_w]

    # Normalize to [0, 1] range
    correlation_cropped = correlation_cropped.astype(np.float32)
    if correlation_cropped.max() > correlation_cropped.min():
        correlation_cropped = (correlation_cropped - correlation_cropped.min()) / \
                              (correlation_cropped.max() - correlation_cropped.min())

    return correlation_cropped

def gradient_correlation(img, kernel):
    """
    Correlation based on gradient magnitude - emphasizes structure over intensity.

    Computes gradients of both image and kernel, then correlates gradient magnitudes.
    This is more robust to illumination changes and emphasizes particle edges/structure.

    Args:
        img: Image (float32)
        kernel: Template kernel (float32)

    Returns:
        Gradient-based correlation map
    """
    # Compute gradients
    img_gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    img_gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    img_grad = np.sqrt(img_gx**2 + img_gy**2)

    kernel_gx = cv2.Sobel(kernel, cv2.CV_32F, 1, 0, ksize=3)
    kernel_gy = cv2.Sobel(kernel, cv2.CV_32F, 0, 1, ksize=3)
    kernel_grad = np.sqrt(kernel_gx**2 + kernel_gy**2)

    # Normalize gradients
    img_grad = img_grad / (img_grad.max() + 1e-10)
    kernel_grad = kernel_grad / (kernel_grad.max() + 1e-10)

    # Correlate gradient magnitudes
    if gpu.USE_OPENCL:
        img_gpu = cv2.UMat(img_grad)
        k_gpu = cv2.UMat(kernel_grad)
        result = cv2.matchTemplate(img_gpu, k_gpu, cv2.TM_CCORR_NORMED)
        return result.get()

    return cv2.matchTemplate(img_grad, kernel_grad, cv2.TM_CCORR_NORMED)

