"""Pruned from the research module ``gpu_backend.py``: only the code reachable from the shipped
detectors (csv, ws_multi_vote / ws_multi_vote_gpu, ws_vgl_log, cellpose) is kept, unchanged.

Original module docstring:

gpu_backend.py - Phase 3d capability probe + GPU helpers.

This module isolates GPU acceleration so the rest of the benchmark can stay
CPU-pure. It detects three independent capabilities at import time:

  1. cv2.cuda compiled-in (the spec's preferred backend)
  2. CuPy core (basic arithmetic, ElementwiseKernel JIT, RawKernel JIT, FFT)
  3. CuPy distance_transform_edt (uses pre-built kernels, separate from
     gaussian_filter / convolve which were broken in our environment)

On the executor host (Windows + CUDA Toolkit 13.0 + opencv-python 4.12 +
CuPy 13.6) we observed:

  - cv2.cuda.getCudaEnabledDeviceCount() == 0 (the pip wheel has no CUDA)
  - CuPy core works (RawKernel, ElementwiseKernel, FFT, DT)
  - cupyx.scipy.ndimage.gaussian_filter / .convolve / .laplace FAIL because
    they JIT-compile against libcudacxx headers that conflict with
    cuda_fp8.hpp from CUDA 13. (See PHASE3D_REPORT for diagnosis.)

So we use:
  - CuPy DT (works)             instead of cv2.distanceTransform / scipy
  - FFT-based Gaussian          instead of cv2.GaussianBlur / cnd.gaussian_filter
  - RawKernel Laplacian         instead of cv2.Laplacian / cnd.laplace

If cv2.cuda becomes available later (e.g. someone installs a CUDA-enabled
opencv-contrib build), the cv2.cuda paths can be added without changing the
public API exposed here.
"""

import numpy as np


def _probe_cv2_cuda() -> bool:
    try:
        import cv2
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (ImportError, AttributeError):
        return False
    except Exception:
        return False

def _probe_cupy_core() -> bool:
    try:
        import cupy as cp  # noqa: F401
        # Touch a tiny array to confirm the runtime is actually working.
        a = cp.asarray(np.zeros(4, dtype=np.float64))
        _ = a + a
        cp.cuda.Stream.null.synchronize()
        return True
    except Exception:
        return False

def _probe_cupy_dt() -> bool:
    """Distance transform uses pre-built kernels (no JIT against CUDA 13 headers)."""
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cnd
        a = cp.zeros((8, 8), dtype=bool); a[2:5, 2:5] = True
        _ = cnd.distance_transform_edt(a)
        cp.cuda.Stream.null.synchronize()
        return True
    except Exception:
        return False

def _probe_cupy_rawkernel() -> bool:
    try:
        import cupy as cp
        krn = cp.RawKernel(r'''
extern "C" __global__ void _probe(double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + 1.0;
}
''', '_probe')
        a = cp.zeros(64, dtype=cp.float64)
        krn((1,), (64,), (a, 64))
        cp.cuda.Stream.null.synchronize()
        return True
    except Exception:
        return False

HAS_CV2_CUDA = _probe_cv2_cuda()

HAS_CUPY = _probe_cupy_core()

HAS_CUPY_DT = HAS_CUPY and _probe_cupy_dt()

HAS_CUPY_RAWKERNEL = HAS_CUPY and _probe_cupy_rawkernel()

# A "GPU-able" backend means we can plausibly accelerate the watershed
# pipeline. We need at minimum CuPy core + DT (the dominant cost in
# classic_watershed). Without DT, GPU brings nothing meaningful here.
GPU_AVAILABLE = HAS_CV2_CUDA or (HAS_CUPY and HAS_CUPY_DT)

_LAP_RAWKERNEL = None  # filled lazily on first use

def _build_lap_rawkernel():
    """Lazily build a 3x3 Laplacian RawKernel that matches cv2.Laplacian ksize=3.

    cv2.Laplacian(ddepth=CV_64F, ksize=3) uses the kernel
        [[2, 0, 2],
         [0,-8, 0],
         [2, 0, 2]]
    (verified empirically against an impulse response). Border handling:
    OpenCV's default is BORDER_REFLECT_101 — pixels outside are mirrored
    around the boundary pixel. We reproduce that by clamping to nearest
    interior index for the 1-pixel border, which is a common simplification
    that introduces at most a 1-pixel difference on the very edge of the
    image; that affects no real particle (which is rmin=6 from any edge).
    """
    import cupy as cp
    src = r'''
extern "C" __global__ void lap3x3_diag(const double* in, double* out,
                                       int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int xm = x > 0 ? x - 1 : x + 1;
    int xp = x < W - 1 ? x + 1 : x - 1;
    int ym = y > 0 ? y - 1 : y + 1;
    int yp = y < H - 1 ? y + 1 : y - 1;
    // Diagonals only: 2*(NW + NE + SW + SE) - 8*center
    double nw = in[ym*W + xm];
    double ne = in[ym*W + xp];
    double sw = in[yp*W + xm];
    double se = in[yp*W + xp];
    double c  = in[y*W + x];
    out[y*W + x] = 2.0 * (nw + ne + sw + se) - 8.0 * c;
}
'''
    return cp.RawKernel(src, 'lap3x3_diag')

def gpu_laplacian_3x3(field_f64_gpu):
    """3x3 Laplacian on GPU matching cv2.Laplacian ksize=3.

    Input/output: cupy float64 (H, W).
    """
    global _LAP_RAWKERNEL
    import cupy as cp
    if _LAP_RAWKERNEL is None:
        _LAP_RAWKERNEL = _build_lap_rawkernel()
    H, W = field_f64_gpu.shape
    out = cp.empty_like(field_f64_gpu)
    threads = (16, 16)
    blocks = ((W + 15) // 16, (H + 15) // 16)
    _LAP_RAWKERNEL(blocks, threads, (field_f64_gpu, out, H, W))
    return out

def gpu_distance_transform_edt(binary_gpu):
    """Distance transform on GPU. Input: cupy bool / uint8. Output: cupy float64."""
    import cupyx.scipy.ndimage as cnd
    return cnd.distance_transform_edt(binary_gpu)

def to_gpu(arr):
    import cupy as cp
    return cp.asarray(arr)

def to_cpu(arr_gpu):
    import cupy as cp
    return cp.asnumpy(arr_gpu)

def gpu_synchronize():
    if HAS_CUPY:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
