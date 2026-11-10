"""T046 (D30, fix B): frame-0 auto-radius. With --detector csv the radii come from the frame-0 CSV rows, never from a
LoG; other detectors use ONE LoG implementation (skimage, the CPU path) on every device, so CPU and GPU make the same
decision. Committed failing before the fix."""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
CPU_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENCV_OPENCL_RUNTIME="disabled", PYTHONHASHSEED="0",
               PYTHONIOENCODING="utf-8")


def test_rule_keeps_defaults_near_primary_scale():
    from cbtv.run_tracker import _auto_radius_rule
    r = _auto_radius_rule(np.full(50, 4.5))
    assert r["changed"] is False and r["scale_ratio"] == pytest.approx(1.0)


def test_rule_small_particles_same_clamps_as_before():
    from cbtv.run_tracker import _auto_radius_rule
    r = _auto_radius_rule(np.full(50, 2.5))          # ratio 0.56 < 0.75
    assert r["changed"] is True
    assert (r["r_min"], r["r_max"]) == (3, 7)          # max(3, int(1.5)), min(12, int(4.5)+1)=5 -> width >= 4 -> 7
    r = _auto_radius_rule(np.linspace(9, 11, 40))     # ratio 2.2 > 1.5
    assert r["changed"] is True and r["r_max"] <= 12


def test_rule_needs_more_than_ten_radii():
    from cbtv.run_tracker import _auto_radius_rule
    assert _auto_radius_rule(np.full(10, 2.5)) is None


def test_non_csv_detectors_use_one_cpu_log(monkeypatch):
    """Whatever --use-gpu / --opt-gpu-blob say, the auto-radius LoG is the CPU (skimage) implementation."""
    import cbtv.run_tracker as rt
    seen = {}

    def fake_log(img, **kw):
        seen.update(kw)
        return np.zeros((0, 3))
    monkeypatch.setattr(rt, "blob_log_gpu_fast", fake_log)
    args = type("A", (), {"detector": "bayes_adapt", "use_gpu": 1, "opt_gpu_blob": 1, "detections_csv": None})()
    radii, src = rt._auto_radius_radii(np.zeros((32, 32, 3), np.uint8), args)
    assert src == "log" and seen.get("use_gpu") is False


def test_csv_radii_come_from_the_csv(tmp_path):
    import cbtv.run_tracker as rt
    det = tmp_path / "d.csv"
    det.write_text("frame,x,y,r,score\n" + "".join(f"0,{10 + i},{20},{2.0 + 0.1 * i},1\n" for i in range(12))
                   + "1,5,5,9.0,1\n")
    args = type("A", (), {"detector": "csv", "use_gpu": 1, "opt_gpu_blob": 1, "detections_csv": str(det)})()
    radii, src = rt._auto_radius_radii(np.zeros((64, 64, 3), np.uint8), args)
    assert src == "csv"
    # frame-0 rows only, rescaled into LoG-radius units
    want = sorted(round((2.0 + 0.1 * i) * rt.CSV_TO_LOG_RADIUS, 6) for i in range(12))
    assert sorted(np.round(radii, 6)) == want


def test_csv_scale_primary_keeps_defaults():
    """P2: a CSV at PRIMARY's true-radius scale (median 6.555 px) maps to the 4.5 px LoG reference (ratio 1.0)."""
    import cbtv.run_tracker as rt
    a = rt._auto_radius_rule(np.full(50, rt.PRIMARY_CSV_MEDIAN_R) * rt.CSV_TO_LOG_RADIUS)
    assert a["changed"] is False and a["scale_ratio"] == pytest.approx(1.0)


def _small_particle_clip(tmp_path, r_draw=3, r_csv=2.5, n=12):
    """256x256 MJPG clip of 40 small discs drifting 1 px/frame + their csv detections (RES_B_4K-like: small r)."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(20, 236, (40, 2))
    cols = rng.integers(60, 255, (40, 3))
    clip = tmp_path / "small.avi"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter.fourcc(*"MJPG"), 10, (256, 256))
    rows = ["frame,x,y,r,score"]
    for f in range(n):
        img = np.zeros((256, 256, 3), np.uint8)
        for (x, y), c in zip(pts, cols, strict=True):
            cv2.circle(img, (int(x + f), int(y)), r_draw, tuple(int(v) for v in c), -1)
            rows.append(f"{f},{int(x + f)}.0,{int(y)}.0,{r_csv},1.0")
        vw.write(img)
    vw.release()
    det = tmp_path / "det.csv"
    det.write_text("\n".join(rows) + "\n")
    return clip, det


def _run(clip, det, out, gpu, env):
    import run_cbtv
    argv = [sys.executable, "-u", "-m", "cbtv.run_tracker", "--input", str(clip), "--output-video", str(out / "t.mp4"),
            "--output-csv", str(out / "tracks.csv"), "--frame-step", "1", "--detector", "csv", "--detections-csv",
            str(det), "--use-gpu", "1" if gpu else "0", "--gpu-device", "0", "--r-min", "6", "--r-max", "10",
            ] + list(run_cbtv.PRESETS["bt"])
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(argv, cwd=str(ROOT), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout, (out / "tracks.csv").read_bytes()


def test_csv_run_logs_csv_source_and_pinned_radii(tmp_path):
    clip, det = _small_particle_clip(tmp_path)
    log, _ = _run(clip, det, tmp_path / "cpu", False, CPU_ENV)
    line = next(ln for ln in log.splitlines() if ln.startswith("[AUTO-RADIUS]"))
    assert "source=csv" in line and "r_min=3, r_max=7" in line, line


@pytest.mark.gpu
def test_csv_auto_radius_cpu_equals_gpu(tmp_path):
    clip, det = _small_particle_clip(tmp_path)
    gpu_env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    _, cpu = _run(clip, det, tmp_path / "cpu", False, CPU_ENV)
    _, gpu = _run(clip, det, tmp_path / "gpu", True, gpu_env)
    assert cpu == gpu
