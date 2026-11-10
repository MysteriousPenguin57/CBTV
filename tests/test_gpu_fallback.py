"""B14 / T029: `--use-gpu 1` on a machine without a usable CUDA device must fall back to CPU (as init_gpu announces)
and give the CPU result, not crash in a later CuPy call. Simulated with CUDA hidden and OpenCL off, so it runs on
any machine (with or without a GPU); the result must equal the golden byte for byte (CPU == GPU goldens since T027)."""
import os
import subprocess

import pytest
from conftest import FIX, ROOT, first_diff_message, ship_argv


@pytest.mark.parametrize("preset,golden", [("ship-v4", "golden_gpu_tracks.csv"), ("bt", "golden_gpu_tracks_bt.csv")])
def test_use_gpu_without_cuda_falls_back_to_cpu(preset, golden, tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", OPENCV_OPENCL_RUNTIME="disabled",
               PYTHONHASHSEED="0")
    env.pop("CBTV_DUMP_C", None)
    env.pop("CPTV_DUMP_C", None)
    r = subprocess.run(ship_argv("synth_io", tmp_path, use_gpu=True, preset=preset), cwd=str(ROOT), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, (
        "tracker crashed with --use-gpu 1 and no CUDA device (rc=%d)\n%s" % (r.returncode, log[-2000:]))
    out, gold = tmp_path / "tracks.csv", FIX / "synth_io" / golden
    assert out.read_bytes() == gold.read_bytes(), (
        "--use-gpu 1 fallback != %s\n%s" % (golden, first_diff_message(out, gold)))


def test_gpu_probe_rejects_a_cuda_runtime_that_cannot_compile(monkeypatch):
    """CuPy installed but NVRTC broken (e.g. a missing nvrtc-builtins DLL): --gpu auto used to pick the GPU because a
    cached kernel passed the probe, then crashed mid-run. The probe must report no working GPU."""
    import run_cbtv
    from cbtv import gpu
    if not gpu.CUPY_AVAILABLE:
        pytest.skip("CuPy not installed")
    from cupy.cuda import compiler

    def broken(*a, **k):
        raise RuntimeError("nvrtc: error: failed to open nvrtc-builtins64_129.dll.")
    monkeypatch.setattr(compiler, "compile_using_nvrtc", broken)
    assert gpu.cupy_works() is False
    assert run_cbtv.gpu_available() is False
