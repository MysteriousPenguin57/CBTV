"""CPU smoke tests for run_cbtv.py plain-English input errors: exit code 2 + one message line,
raised before any tracking starts. Tiny synthetic inputs, no detector or tracker run."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run_cbtv.py"


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    p = tmp_path_factory.mktemp("vid") / "tiny.avi"
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for i in range(6):
        fr = np.zeros((64, 64, 3), np.uint8)
        cv2.circle(fr, (20 + i, 30), 7, (0, 0, 255), -1)
        wr.write(fr)
    wr.release()
    return p


def cli(*args, env_extra=None, cwd=None):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(RUN), *map(str, args)], capture_output=True,
                          text=True, env=env, cwd=cwd, timeout=120)


def write_csv(path, header, rows):
    path.write_text(header + "\n" + "".join(",".join(map(str, r)) + "\n" for r in rows))
    return path


def assert_err(res, needle, tmp_path):
    assert res.returncode == 2, (res.returncode, res.stdout, res.stderr)
    assert needle in res.stderr, res.stderr
    assert "Traceback" not in res.stderr
    assert not (tmp_path / "out" / "tracks.csv").exists()


def test_missing_video(tmp_path):
    res = cli(tmp_path / "nope.mp4", "--output-dir", tmp_path / "out")
    assert_err(res, "input video not found", tmp_path)


def test_unreadable_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    res = cli(bad, "--output-dir", tmp_path / "out")
    assert_err(res, "cannot read video", tmp_path)


def test_csv_detector_without_csv(tmp_path, video):
    res = cli(video, "--detector", "csv", "--gpu", "0", "--output-dir", tmp_path / "out")
    assert_err(res, "--detector csv needs --detections-csv", tmp_path)


def test_csv_not_found(tmp_path, video):
    res = cli(video, "--detector", "csv", "--detections-csv", tmp_path / "x.csv", "--gpu", "0",
              "--output-dir", tmp_path / "out")
    assert_err(res, "detections CSV not found", tmp_path)


def test_csv_missing_columns(tmp_path, video):
    c = write_csv(tmp_path / "d.csv", "frame,x,y", [(0, 20, 30)])
    res = cli(video, "--detector", "csv", "--detections-csv", c, "--gpu", "0", "--output-dir", tmp_path / "out")
    assert_err(res, "missing column(s) r", tmp_path)


def test_csv_unparseable(tmp_path, video):
    c = write_csv(tmp_path / "d.csv", "frame,x,y,r", [("zero", 20, 30, 7)])
    res = cli(video, "--detector", "csv", "--detections-csv", c, "--gpu", "0", "--output-dir", tmp_path / "out")
    assert_err(res, "cannot parse detections CSV", tmp_path)


def test_explicit_other_detector_with_csv(tmp_path, video):
    c = write_csv(tmp_path / "d.csv", "frame,x,y,r", [(0, 20, 30, 7)])
    res = cli(video, "--detector", "ws_multi_vote", "--detections-csv", c, "--gpu", "0",
              "--output-dir", tmp_path / "out")
    assert_err(res, "--detections-csv only works with --detector csv", tmp_path)
    assert "--detector ws_multi_vote" in res.stderr


def test_zero_detections_first_frames(tmp_path, video):
    c = write_csv(tmp_path / "d.csv", "frame,x,y,r,score", [(7, 20, 30, 7, 1.0), (8, 21, 30, 7, 1.0)])
    res = cli(video, "--detector", "csv", "--detections-csv", c, "--gpu", "0", "--output-dir", tmp_path / "out")
    assert_err(res, "no detections in the first 5 processed frames", tmp_path)


def test_radius_mismatch_suggests_range(tmp_path, video):
    c = write_csv(tmp_path / "d.csv", "frame,x,y,r", [(f, 20 + f, 30, 20.0) for f in range(6)])
    res = cli(video, "--detector", "csv", "--detections-csv", c, "--gpu", "0", "--output-dir", tmp_path / "out")
    assert_err(res, "median detection radius 20.0 px", tmp_path)
    assert "--r-min 15 --r-max 26" in res.stderr


def test_gpu1_without_cupy(tmp_path, video):
    # Hiding every CUDA device makes the CuPy probe fail even where CuPy is installed.
    res = cli(video, "--gpu", "1", "--output-dir", tmp_path / "out", env_extra={"CUDA_VISIBLE_DEVICES": "-1"})
    assert_err(res, 'pip install "cbtv[gpu]"', tmp_path)


def test_verbose_shows_traceback(tmp_path):
    res = cli(tmp_path / "nope.mp4", "--verbose", "1", "--output-dir", tmp_path / "out")
    assert res.returncode != 0 and "Traceback" in res.stderr and "input video not found" in res.stderr


def test_version():
    res = cli("--version")
    assert res.returncode == 0 and res.stdout.startswith("cbtv ")
