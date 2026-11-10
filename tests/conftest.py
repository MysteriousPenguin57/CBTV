"""Shared test plumbing: repo root on sys.path, GPU detection, fixture runner."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
FIXTURES = {"primary60": "clip.mp4", "m05_30": "clip.mp4", "synth_io": "clip.avi"}
for p in (str(ROOT), str(ROOT / "tools" / "parity")):
    if p not in sys.path:
        sys.path.insert(0, p)


def have_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if have_gpu():
        return
    skip = pytest.mark.skip(reason="no CuPy/CUDA device: GPU test NOT run (not a pass)")
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)


def ship_argv(fixture, out_dir, use_gpu, preset="ship-v4"):
    """Tracker argv for a fixture: the preset's flags straight from run_cbtv.PRESETS. The default
    ship-v4 (= run_cbtv.SHIP_CONFIG_V4) produced golden_gpu_tracks.csv (same flags as
    tools/run_fixtures.py); bt produces golden_gpu_tracks_bt.csv."""
    import run_cbtv
    clip = FIX / fixture / FIXTURES[fixture]
    return [sys.executable, "-u", "-m", "cbtv.run_tracker", "--input", str(clip),
            "--output-video", str(out_dir / "tracks.mp4"), "--output-csv", str(out_dir / "tracks.csv"),
            "--frame-step", "1", "--detector", "csv", "--detections-csv", str(FIX / fixture / "detections.csv"),
            "--use-gpu", "1" if use_gpu else "0", "--gpu-device", "0", "--r-min", "6", "--r-max", "10"] \
        + list(run_cbtv.PRESETS[preset])


def run_fixture(fixture, out_dir, use_gpu, dump_dir=None, preset="ship-v4"):
    """Run the tracker on a fixture; return tracks.csv path. CPU runs use the manual determinism env
    (CUDA hidden, OpenCL off, PYTHONHASHSEED=0) until WS-P makes it automatic."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("CBTV_DUMP_C", None)
    env.pop("CPTV_DUMP_C", None)
    if not use_gpu:
        env.update(CUDA_VISIBLE_DEVICES="-1", OPENCV_OPENCL_RUNTIME="disabled", PYTHONHASHSEED="0")
    if dump_dir:
        env["CBTV_DUMP_C"] = str(dump_dir)
    r = subprocess.run(ship_argv(fixture, out_dir, use_gpu, preset), cwd=str(ROOT), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert r.returncode == 0, "tracker failed rc=%d\n%s" % (r.returncode, r.stdout.decode("utf-8", "replace")[-2000:])
    try:
        (out_dir / "tracks.mp4").unlink()
    except OSError:
        pass
    return out_dir / "tracks.csv"


def first_diff_message(a_path, b_path):
    """Readable first differing row of two csv files (for assertion messages)."""
    la = Path(a_path).read_text().splitlines()
    lb = Path(b_path).read_text().splitlines()
    for i, (x, y) in enumerate(zip(la, lb, strict=False)):
        if x != y:
            return "first differing row %d:\n  A: %s\n  B: %s" % (i, x, y)
    return "row counts differ: %d vs %d" % (len(la), len(lb))


def pytest_assertrepr_compare(config, op, left, right):
    """Short explanation for == on large bytes/str (golden csv compares). pytest's default renders a
    difflib ndiff of the two sequences, which on a ~1 MB mismatch runs for tens of minutes (seen on
    macOS arm64 CI) so the test never reports. The assertion itself is unchanged."""
    if op != "==" or type(left) is not type(right) or not isinstance(left, (bytes, str)):
        return None
    if max(len(left), len(right)) < 4096:
        return None
    i = next((k for k, (a, b) in enumerate(zip(left, right, strict=False)) if a != b), min(len(left), len(right)))
    nl = b"\n" if isinstance(left, bytes) else "\n"
    row = left[:i].count(nl)
    return ["large %s differ: len %d vs %d, first difference at offset %d (row %d)"
            % (type(left).__name__, len(left), len(right), i, row)]
