"""Regression tests for bugs found during the release review (ids as in the release bug list).

Each test was committed failing before its fix."""
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENCV_OPENCL_RUNTIME="disabled", PYTHONHASHSEED="0",
               PYTHONIOENCODING="utf-8")


def _small_clip(tmp_path, size=128, n_frames=6, r=7):
    """A size x size MJPG clip with 4 bright discs of radius r drifting 1 px/frame, plus their csv detections."""
    clip = tmp_path / "small.avi"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (size, size))
    rows = ["frame,x,y,r,score"]
    starts = [(30, 30), (90, 30), (30, 90), (90, 90)]
    for f in range(n_frames):
        img = np.zeros((size, size, 3), np.uint8)
        for k, (x0, y0) in enumerate(starts):
            x, y = x0 + f, y0
            cv2.circle(img, (x, y), r, (60 + 40 * k, 200, 255 - 40 * k), -1)
            rows.append(f"{f},{x}.0,{y}.0,{r}.0,1.0")
        vw.write(img)
    vw.release()
    det = tmp_path / "det.csv"
    det.write_text("\n".join(rows) + "\n")
    return clip, det


def test_b07_small_image_keeps_csv_radii(tmp_path):
    """B07: min(H, W) < 256 used to force r_min=2 / r_max=4 at the default radii and drop csv detections with r~7."""
    clip, det = _small_clip(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "run_cbtv.py"), "--input", str(clip), "--detector", "csv",
                        "--detections-csv", str(det), "--gpu", "0", "--output-dir", str(out)],
                       cwd=str(tmp_path), env=CPU_ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    import pandas as pd
    tr = pd.read_csv(out / "tracks.csv")
    assert len(tr) > 0, "no tracked rows: the csv detections (r=7) were dropped"
    # every disc is a csv detection in every frame: with the forced r_min=2 / r_max=4 most were demoted to soft or
    # deduplicated, so tracks coasted (missed > 0)
    assert (tr["missed"] == 0).all(), tr[tr["missed"] > 0].to_string()


def test_b07_small_image_warns_on_radius_override(tmp_path):
    """B07: a non-csv detector on a small image still gets the rescaled radii, but no longer silently."""
    clip, _ = _small_clip(tmp_path, n_frames=2)
    r = subprocess.run([sys.executable, "-u", "-m", "cbtv.run_tracker", "--input", str(clip),
                        "--output-video", str(tmp_path / "o.mp4"), "--output-csv", str(tmp_path / "t.csv"),
                        "--use-gpu", "0", "--max-frames", "1", "--dump-first-frame-corr", "0"],
                       cwd=str(ROOT), env=CPU_ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "[WARN] small-image mode" in r.stdout


def test_b10_plain_run_writes_no_debug_dir(tmp_path):
    """B10: a plain `cbtv` run (no --verbose) used to write debug_hybrid/ into --output-dir."""
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "run_cbtv.py"), "--input", str(FIX / "synth_io" / "clip.avi"),
                        "--detector", "csv", "--detections-csv", str(FIX / "synth_io" / "detections.csv"),
                        "--gpu", "0", "--max-frames", "2", "--output-dir", str(out)],
                       cwd=str(tmp_path), env=CPU_ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / "tracks.csv").is_file()
    assert not (out / "debug_hybrid").exists()
    assert not (tmp_path / "debug_hybrid").exists()


def test_b12_ws_vgl_log_without_gpu_fails_loudly():
    """B12: ws_vgl_log_gpu has no CPU path; without CuPy it used to return 0 detections on every frame, silently."""
    code = ("import sys, cv2; sys.path.insert(0, r'%s'); from cbtv import ws_detector\n"
            "from cbtv.detect import gpu_backend as gb\n"
            "if gb.HAS_CUPY and gb.HAS_CUPY_RAWKERNEL: print('HAS_GPU'); sys.exit(0)\n"
            "cap = cv2.VideoCapture(r'%s'); ok, fr = cap.read()\n"
            "try:\n    ws_detector.detect_ws(fr, 6, 10, 'ws_vgl_log_gpu')\nexcept RuntimeError as e:\n"
            "    print('RAISED', e); sys.exit(0)\nprint('SILENT'); sys.exit(1)\n"
            % (ROOT, FIX / "primary60" / "clip.mp4"))
    r = subprocess.run([sys.executable, "-c", code], env=CPU_ENV, capture_output=True, text=True)
    if "HAS_GPU" in r.stdout:
        return  # the GPU path exists here; the CPU-only behaviour is what this test pins
    assert r.returncode == 0 and "RAISED" in r.stdout, r.stdout + r.stderr
    assert "ws_vgl_log" in r.stdout and "GPU" in r.stdout and "ws_multi_vote" in r.stdout


def test_b13_test_extra_declared_and_used_by_ci():
    """B13: tests import pandas (and hypothesis) but no extra declared them; CI installed them by hand."""
    # no tomllib on py3.10: the [test] line of [project.optional-dependencies] is a one-line list
    m = re.search(r"^test\s*=\s*\[([^\]]*)\]", (ROOT / "pyproject.toml").read_text(), re.M)
    test_extra = (m.group(1) if m else "").lower()
    for dep in ("pytest", "pandas", "hypothesis"):
        assert dep in test_extra, f"[test] extra lacks {dep}"
    for wf in ("ci.yml", "ci-windows.yml", "ci-macos.yml"):
        text = (ROOT / ".github" / "workflows" / wf).read_text()
        assert "pytest pandas hypothesis" not in text, f"{wf} still installs the test deps by hand"
        assert ".[test]" in text, f"{wf} does not install the [test] extra"


def test_b06_second_pass_runs(tmp_path):
    """B06: --second-pass 1 called compute_cost_matrix_gpu without frame_lab, so every run with it raised TypeError."""
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "run_cbtv.py"), "--input", str(FIX / "synth_io" / "clip.avi"),
                        "--detector", "csv", "--detections-csv", str(FIX / "synth_io" / "detections.csv"),
                        "--gpu", "0", "--max-frames", "15", "--output-dir", str(out), "--extra", "--second-pass 1"],
                       cwd=str(tmp_path), env=CPU_ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (out / "tracks.csv").is_file()
