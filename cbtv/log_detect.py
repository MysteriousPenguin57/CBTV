"""GPU-accelerated Laplacian-of-Gaussian blob detection."""
import cv2
import numpy as np

from . import gpu
from .gpu import cp


def _generate_log_kernel(sigma, size=None):
    """Generate a normalised Laplacian-of-Gaussian (LoG) kernel.

    Computes the analytically exact LoG filter kernel at a given sigma
    and crops it to an odd-sized square whose side length defaults to
    ``ceil(6 * sigma)`` pixels.  The mean is subtracted so the kernel
    integrates to zero, which prevents DC offsets from corrupting the
    filter response when convolved with bright flat regions.

    Pipeline context:
        Called by: blob_log_gpu_fast
        Calls: (NumPy only)
        Stage: detection

    Args:
        sigma: Gaussian scale parameter in pixels.  Larger sigma detects
            larger blobs; the optimal sigma for a blob of radius r is
            approximately ``r / sqrt(2)``. [float]
        size: Explicit kernel side length (must be odd).  If None, the
            kernel is auto-sized to ``ceil(6 * sigma)`` rounded up to the
            next odd integer. [int or None]

    Returns:
        2-D float64 NumPy array of shape ``(size, size)`` containing the
        mean-subtracted LoG kernel. [np.ndarray]

    Phase history:
        - Phase 3: New function added alongside GPU LoG detection.
    """
    import math
    if size is None:
        size = int(math.ceil(sigma * 6))
        if size % 2 == 0: size += 1
    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    xx, yy = np.meshgrid(x, y)
    kernel = -(1.0 / (np.pi * sigma**4)) * (1.0 - (xx**2 + yy**2) / (2 * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel - kernel.mean()
    return kernel

def _simple_nms_gpu(blobs_gpu, threshold=3.0):
    """GPU-assisted non-maximum suppression for LoG blob candidates.

    Transfers the CuPy blob array to CPU, sorts candidates by LoG
    response magnitude in descending order, then greedily suppresses any
    candidate whose centre lies within ``threshold`` pixels of a
    stronger, already-kept candidate.  Only the first three columns
    (y, x, sigma) are returned; the response-magnitude column 3 is
    dropped.

    Falls back to returning the raw input unchanged when CuPy is not
    available (e.g. CPU-only mode or after LoG skimage fallback).

    Pipeline context:
        Called by: blob_log_gpu_fast
        Calls: (NumPy only after GPU transfer)
        Stage: detection

    Args:
        blobs_gpu: CuPy array of shape ``(N, 4)`` with columns
            ``[y, x, sigma, response]`` as produced by the scale-space
            maximum search in ``blob_log_gpu_fast``. [cp.ndarray]
        threshold: Minimum centre-to-centre pixel distance below which
            a weaker candidate is suppressed.  Defaults to
            ``min_sigma * 1.5`` as passed by ``blob_log_gpu_fast``.
            [float]

    Returns:
        NumPy array of shape ``(M, 3)`` with columns ``[y, x, sigma]``
        for the M surviving blob candidates. [np.ndarray]

    Phase history:
        - Phase 3: New function added alongside GPU LoG detection.
    """
    if not gpu.USE_CUPY:
        return blobs_gpu
    blobs = blobs_gpu.get()
    if len(blobs) == 0: return blobs[:, :3]
    # Sort by response magnitude descending (column 3)
    blobs = blobs[np.argsort(-blobs[:, 3], kind="stable")]
    keep = np.ones(len(blobs), dtype=bool)
    for i in range(len(blobs)):
        if not keep[i]: continue
        y1, x1, s1, _ = blobs[i]
        dists = np.sqrt((blobs[:, 0] - y1)**2 + (blobs[:, 1] - x1)**2)
        suppress = (dists < threshold) & (np.arange(len(blobs)) > i)
        keep[suppress] = False

    return blobs[keep, :3]

def blob_log_gpu_fast(image, min_sigma=3.5, max_sigma=6.5, num_sigma=4, threshold=0.08, use_gpu=True):
    # [Phase 3] GPU LoG detection — supplements Hough in dense regions (see Flowchart 6)
    """GPU-accelerated Laplacian-of-Gaussian blob detector.

    Builds a ``num_sigma``-level LoG scale space over ``[min_sigma,
    max_sigma]``.  Each level is convolved using OpenCV UMat (OpenCL) to
    avoid CuPy filter-compilation overhead, scaled by ``sigma^2`` for
    size-normalised response, then the 3-D array is searched for local
    maxima across scale and space.  Surviving peaks above the scaled
    threshold are de-duplicated by ``_simple_nms_gpu``.

    Falls back to ``skimage.feature.blob_log`` when GPU is unavailable.

    Used as the primary exemplar generator when Hough produces fewer
    than 530 candidates (supplement mode) or fewer than 400 (full
    replace mode).  Provides sharper exemplars in dense overlap regions
    where Hough kernels become "poisoned" by neighbouring particles.

    Pipeline context:
        Called by: user_detection_with_bayes_adapt (LoG branch)
        Calls: _generate_log_kernel, _simple_nms_gpu
        Stage: detection

    Args:
        image: Single-channel float32 or uint8 grayscale frame, shape
            ``(H, W)``. [np.ndarray]
        min_sigma: Smallest LoG scale in pixels (lower bound of sigma
            sweep). [float]
        max_sigma: Largest LoG scale in pixels (upper bound of sigma
            sweep). [float]
        num_sigma: Number of evenly spaced sigma values in
            ``[min_sigma, max_sigma]``. [int]
        threshold: Minimum normalised LoG response to accept a blob.
            Internally scaled by ``255 * 2.4`` to match pixel magnitude.
            [float]
        use_gpu: If False, or if CuPy is unavailable, falls back to
            skimage CPU implementation. [bool]

    Returns:
        NumPy array of shape ``(N, 3)`` with columns ``[y, x, sigma]``
        for each detected blob centre. [np.ndarray]

    Phase history:
        - Phase 3: New function replacing Hough in dense-overlap frames.
    """
    if not (use_gpu and gpu.USE_CUPY):
        from skimage.feature import blob_log
        return blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

    # [LoG-1] Image preprocessing and normalization
    sigmas = cp.linspace(min_sigma, max_sigma, num_sigma)
    H, W = image.shape

    scale_space = cp.empty((num_sigma, H, W), dtype=cp.float32)
    img_umat = cv2.UMat(image) # UMat runs on OpenCL nicely, avoiding cupy filter compilation bugs

    # [LoG-2] Scale-space stack construction (σ grid)
    for i, s in enumerate(sigmas.get()):
        k_cpu = _generate_log_kernel(s)
        out_umat = cv2.filter2D(img_umat, cv2.CV_32F, k_cpu)
        resp = cp.asarray(out_umat.get())
        scale_space[i] = -resp * (s ** 2)
    print(f"[BLOB_DIAG] scale_space max={scale_space.max()} mean={scale_space.mean()} ss={scale_space.shape}")

    # [LoG-3] 3D maximum filter + threshold gate
    # Compute local maxima without wrapping around.
    # Instead of cp.roll and keeping the wrap, we zero out the wrapped edge.
    roll_z1 = cp.roll(scale_space, 1, axis=0); roll_z1[0, :, :] = 0
    roll_z_minus1 = cp.roll(scale_space, -1, axis=0); roll_z_minus1[-1, :, :] = 0
    max_z = cp.maximum(scale_space, roll_z1)
    max_z = cp.maximum(max_z, roll_z_minus1)

    roll_y1 = cp.roll(max_z, 1, axis=1); roll_y1[:, 0, :] = 0
    roll_y_minus1 = cp.roll(max_z, -1, axis=1); roll_y_minus1[:, -1, :] = 0
    max_zy = cp.maximum(max_z, roll_y1)
    max_zy = cp.maximum(max_zy, roll_y_minus1)

    roll_x1 = cp.roll(max_zy, 1, axis=2); roll_x1[:, :, 0] = 0
    roll_x_minus1 = cp.roll(max_zy, -1, axis=2); roll_x_minus1[:, :, -1] = 0
    max_zyx = cp.maximum(max_zy, roll_x1)
    max_zyx = cp.maximum(max_zyx, roll_x_minus1)

    scaled_threshold = threshold * 255.0 * 2.4
    is_max = (scale_space == max_zyx) & (scale_space > scaled_threshold)

    # Do not explicitly suppress boundaries anymore, since we handled them gracefully
    # Except maybe image spatial boundaries if desired
    is_max[:, 0, :] = False; is_max[:, -1, :] = False
    is_max[:, :, 0] = False; is_max[:, :, -1] = False

    # [LoG-4] NMS / peak extraction
    z, y, x = cp.where(is_max)
    resp_vals = scale_space[z, y, x]
    blobs = cp.stack((y, x, sigmas[z], resp_vals), axis=1)

    # [LoG-5] Return packaging (centers, radii, responses)
    # [v6 Change] blob_log_gpu_fast is entirely v6-new — no original equivalent.
    # Generates LoG exemplars on GPU to supplement/replace Hough in dense regions.
    return _simple_nms_gpu(blobs, threshold=min_sigma*1.5)

