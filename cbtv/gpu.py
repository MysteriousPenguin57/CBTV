"""GPU infrastructure: device init, array helpers, shared backend flags.

Module-level flags are the single source of truth for GPU state; other modules
read them as ``gpu.USE_CUPY`` so init_gpu's runtime mutation is visible.
"""
import cv2
import numpy as np

USE_CUPY = False
USE_OPENCL = False
CUPY_AVAILABLE = False
OPENCL_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.ndimage  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    cp = None

_PROBE_SRC = 'extern "C" __global__ void cbtv_probe(float* a) { a[0] = 1.0f; }'


def cupy_works():
    """True if CuPy can allocate, reduce and compile a CUDA kernel on the current device.

    The compile goes through NVRTC directly, without CuPy's kernel cache, so an incomplete CUDA runtime (for example
    a missing nvrtc-builtins DLL) is caught here rather than in the middle of a run, where a cached kernel would
    otherwise hide it until the first uncached one."""
    if not CUPY_AVAILABLE:
        return False
    try:
        float(cp.zeros(1).sum())
        from cupy.cuda import compiler
        compiler.compile_using_nvrtc(_PROBE_SRC)
        return True
    except Exception:
        return False


def init_gpu(use_gpu, gpu_device=None):
    """Initialize GPU acceleration with smart fallback

    Args:
        use_gpu: Enable GPU acceleration
        gpu_device: Specific GPU device ID to use (0, 1, etc.). If None, uses default.
    """
    global USE_CUPY, USE_OPENCL, OPENCL_AVAILABLE

    if not use_gpu:
        # cv2 otherwise routes UMat ops (log_detect) through OpenCL even on a
        # CPU run; turn it off so --use-gpu 0 is CPU-only without env vars.
        cv2.ocl.setUseOpenCL(False)
        print("[GPU] GPU acceleration disabled - using CPU only (OpenCL off)")
        return

    # Try OpenCL (best for cv2.matchTemplate)
    try:
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            OPENCL_AVAILABLE = True
            USE_OPENCL = True
            print(f"[GPU] OpenCL enabled - Device: {cv2.ocl.Device.getDefault().name()}")
        else:
            print("[GPU] OpenCL not available")
    except Exception as e:
        print(f"[GPU] OpenCL failed: {e}")
    if not CUPY_AVAILABLE:
        print("[WARN] GPU requested but CuPy is not installed: CuPy stages run on CPU. "
              "Install cupy-cuda12x for GPU.")
    if CUPY_AVAILABLE:
        try:
            # Set specific GPU device if requested
            if gpu_device is not None:
                cp.cuda.Device(gpu_device).use()
                print(f"[GPU] Explicitly selecting GPU device {gpu_device}")

            if not cupy_works():
                raise RuntimeError("CuPy cannot run or compile CUDA kernels (check the CUDA runtime and CUDA_PATH)")
            USE_CUPY = True
            device = cp.cuda.Device()
            # FIX: Handle device name properly
            try:
                name = device.attributes['Name']
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                print(f"[GPU] CuPy enabled - GPU: {name}")
            except Exception:
                print(f"[GPU] CuPy enabled - GPU device {device.id}")
        except Exception as e:
            USE_CUPY = False
            print(f"[GPU] CuPy failed to initialize: {e}")

    if not USE_CUPY and not USE_OPENCL:
        print("[GPU] No GPU acceleration available - falling back to CPU")

def xp():
    """Get array module (cupy or numpy)"""
    return cp if USE_CUPY else np

def to_numpy(arr):
    """Convert to numpy if needed"""
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def to_gpu(arr):
    """Convert to GPU if CuPy enabled"""
    if USE_CUPY and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr

