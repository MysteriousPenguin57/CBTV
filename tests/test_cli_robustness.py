"""C013: CLI robustness. Every user error -> exit code 2 + one `[cbtv] error:` line, no traceback, nothing tracked.

In-process Hypothesis properties for the input checks (fast), plus one subprocess run of the real CLI per case."""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_cbtv  # noqa: E402

RUN = ROOT / "run_cbtv.py"
GOOD = "frame,x,y,r,score\n" + "".join(f"{t},{20 + t},30,7,1\n" for t in range(6))

# ---------------------------------------------------------------- in-process properties
token = st.one_of(st.integers(-5, 50).map(str), st.floats(allow_nan=True, allow_infinity=True).map(repr),
                  st.sampled_from(["", "nan", "inf", "-inf", "NaN", "1e309", "abc", " 7 ", "7.0", "-0", "é", "\x00"]))
row = st.lists(token, min_size=0, max_size=6).map(",".join)
header = st.sampled_from(["frame,x,y,r,score", "frame,x,y,r", "frame,x,y", "x,y,r", "", "frame,x,y,r,score,colour"])


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(header, st.lists(row, max_size=12))
def test_csv_check_only_raises_usererror(tmp_path, head, rows):
    p = tmp_path / "d.csv"
    p.write_text(head + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    try:
        run_cbtv.check_csv_detections(str(p), 6, 10)
    except run_cbtv.UserError as e:
        assert "\n" not in str(e)
        return
    # accepted -> every row the tracker will read is finite with r > 0
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vals = [float(r[k]) for k in ("x", "y", "r")]
            assert all(math.isfinite(v) for v in vals) and vals[2] > 0
            int(float(r["frame"]))


def ns(**kw):
    base = dict(frame_step=1, max_frames=-1, r_min=6, r_max=10, detector="csv", output_dir="results", extra="")
    base.update(kw)
    return argparse.Namespace(**base)


@settings(max_examples=300, deadline=None)
@given(st.integers(-10**9, 10**9), st.integers(-10**9, 10**9), st.integers(-50, 200), st.integers(-50, 200))
def test_check_args_ranges(tmp_path_factory, step, maxf, rmin, rmax):
    a = ns(frame_step=step, max_frames=maxf, r_min=rmin, r_max=rmax)
    valid = step >= 1 and (maxf == -1 or maxf >= 1) and 1 <= rmin <= rmax
    if valid:
        run_cbtv.check_args(a)
    else:
        with pytest.raises(run_cbtv.UserError):
            run_cbtv.check_args(a)


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=40))
def test_check_args_extra_never_tracebacks(extra):
    try:
        run_cbtv.check_args(ns(extra=extra))
    except run_cbtv.UserError:
        pass


# ---------------------------------------------------------------- real CLI, one subprocess per case
@pytest.fixture(scope="module")
def work(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    d = tmp_path_factory.mktemp("cli")
    wr = cv2.VideoWriter(str(d / "tiny.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for i in range(6):
        fr = np.zeros((64, 64, 3), np.uint8)
        cv2.circle(fr, (20 + i, 30), 7, (0, 0, 255), -1)
        wr.write(fr)
    wr.release()
    (d / "afile").write_text("x")
    return d


def cli(work, name, args, video=None):
    out = work / ("out_" + name)
    argv = [sys.executable, str(RUN), str(video or work / "tiny.avi"), "--gpu", "0"]
    if "--output-dir" not in args:
        argv += ["--output-dir", str(out)]
    r = subprocess.run(argv + [str(a) for a in args], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1"))
    return r, out


def csv_file(work, name, text):
    p = work / name
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding="utf-8")
    return p


ERROR_CASES = {  # name: (csv text or None, extra args, expected message part)
    "empty_csv": ("", [], "missing column(s)"),
    "header_only": ("frame,x,y,r\n", [], "no detections in the first"),
    "nan_r": (GOOD.replace(",7,", ",nan,", 1), [], "must be finite"),
    "nan_x": (GOOD.replace("20,30", "nan,30"), [], "must be finite"),
    "inf_y": (GOOD.replace(",30,", ",inf,", 1), [], "must be finite"),
    "negative_r": (GOOD.replace(",7,", ",-7,", 1), [], "radius r must be > 0"),
    "zero_r": (GOOD.replace(",7,", ",0,", 1), [], "radius r must be > 0"),
    "blank_frame": (GOOD.replace("0,20", ",20", 1), [], "cannot parse detections CSV"),
    "short_row": (GOOD + "3,20\n", [], "has too few columns"),
    "inf_frame": (GOOD + "inf,20,30,7,1\n", [], "cannot parse detections CSV"),
    # the platform's default text encoding decides: utf-8 fails to decode, cp1252 (Windows) decodes to junk columns
    "binary_csv": (b"\xff\xfe\x00\x01garbage\x80\n", [], ("cannot parse detections CSV", "missing column(s)")),
    "frame_step_0": (GOOD, ["--frame-step", "0"], "--frame-step must be 1 or more"),
    "frame_step_negative": (GOOD, ["--frame-step", "-2"], "--frame-step must be 1 or more"),
    "frame_step_text": (GOOD, ["--frame-step", "two"], "invalid int value"),
    "max_frames_0": (GOOD, ["--max-frames", "0"], "--max-frames must be -1"),
    "max_frames_negative": (GOOD, ["--max-frames", "-5"], "--max-frames must be -1"),
    "unknown_preset": (GOOD, ["--preset", "nope"], "invalid choice"),
    "unknown_detector": (None, ["--detector", "nope"], "unknown --detector"),
    "r_min_gt_r_max": (GOOD, ["--r-min", "12", "--r-max", "6"], "--r-min <= --r-max"),
    "r_min_0": (GOOD, ["--r-min", "0"], "--r-min <= --r-max"),
    "extra_unbalanced_quote": (GOOD, ["--extra", '--foo "bar'], "cannot parse --extra"),
}


@pytest.mark.parametrize("name", sorted(ERROR_CASES))
def test_user_error_exit2_one_line(work, name):
    text, args, needle = ERROR_CASES[name]
    if text is not None:
        args = ["--detections-csv", csv_file(work, name + ".csv", text)] + args
    r, out = cli(work, name, args)
    assert r.returncode == 2, (r.returncode, r.stderr[-2000:])
    lines = r.stderr.strip().splitlines()
    needles = needle if isinstance(needle, tuple) else (needle,)
    assert len(lines) == 1 and lines[0].startswith("[cbtv] error:") and any(n in lines[0] for n in needles), r.stderr
    assert "Traceback" not in r.stderr + r.stdout
    assert not (out / "tracks.csv").exists()


def test_csv_is_a_directory(work):
    r, _ = cli(work, "csv_dir", ["--detections-csv", work])
    assert r.returncode == 2 and "cannot parse detections CSV" in r.stderr and "Traceback" not in r.stderr


def test_video_is_a_directory(work):
    r, _ = cli(work, "video_dir", ["--detections-csv", csv_file(work, "ok.csv", GOOD)], video=work)
    assert r.returncode == 2 and "cannot read video" in r.stderr and "Traceback" not in r.stderr


def test_output_dir_is_a_file(work):
    r, _ = cli(work, "outfile", ["--detections-csv", csv_file(work, "ok.csv", GOOD), "--output-dir", work / "afile"])
    assert r.returncode == 2 and "is not a directory" in r.stderr and "Traceback" not in r.stderr


@pytest.mark.parametrize("name,args", [
    ("non_ascii_paths", None),
    ("frame_step_huge", ["--frame-step", "1000000"]),
    ("max_frames_huge", ["--max-frames", "999999999"]),
    ("negative_frame_row", "neg"),
])
def test_valid_edge_cases_run(work, name, args):
    video = None
    text = GOOD
    if name == "non_ascii_paths":
        d = work / "vidé ö 名前"
        d.mkdir(exist_ok=True)
        video = d / "clip ü.avi"
        video.write_bytes((work / "tiny.avi").read_bytes())
        args = []
    elif args == "neg":
        text, args = GOOD + "-3,20,30,7,1\n", []
    r, out = cli(work, name, ["--detections-csv", csv_file(work, name + ".csv", text)] + args, video=video)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "tracks.csv").exists()
