#!/usr/bin/env python
"""run_cbtv.py — run the CBTV colour particle tracker on an experimental video.

One command in, four outputs out:

  results/tracks.csv       frame, track_id, x, y, r, missed   (the data)
  results/annotated.mp4    every particle circled in a persistent per-track
                           colour with its track ID drawn next to it
  results/overlay.mp4      the tracker's own diagnostic overlay
                           (green = raw detections, orange = live tracks,
                           grey = coasting)
  results/summary.txt      track counts, lengths, coverage statistics

Quick start (GPU if available, else CPU):

  cbtv my_experiment.mp4                (after pip install .)
  python run_cbtv.py my_experiment.mp4  (from a checkout)

Common options:

  --output-dir results     where everything is written
  --detector ws_multi_vote detection preset (see docs/user_guide.md)
  --gpu auto|1|0           auto tries CuPy/CUDA and falls back to CPU
  --r-min 6 --r-max 10     expected particle radius range in PIXELS
  --max-frames N           only process the first N frames (quick test)
  --frame-step N           process every Nth frame
  --label id|none          annotation label style
  --extra "..."            extra raw cbtv.run_tracker flags (advanced)

The default tracking configuration is the `bt` preset (constellation
matching, a cluster budget scaled with particle count, a PIV motion prior and
graph-structural matching, plus edge and coasting handling). See
docs/user_guide.md for every option, the defaults and tuning advice.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# CBTV published v4 ship configuration ("capauto"): constellations + adaptive
# per-N cluster cap + PIV motion prior + graph-structural matching. Reproduces
# the master_table_v4 numbers; preset `ship-v4`.
SHIP_CONFIG_V4 = [
    "--use-constellations", "1", "--cluster-rounds", "2",
    "--cluster-max-per-frame", "256", "--cluster-cap-auto", "1",
    "--cluster-cap-alpha", "0.324", "--cluster-cap-max", "2048",
    "--mu-source", "piv", "--piv-max-disp-ratio", "1.0",
    "--match-op", "graph-struct", "--match-gate-d", "1.1",
    "--w-kernel", "0.1", "--w-dist", "0.8", "--w-color", "0.1",
    "--use-affine-field", "0", "--opt-predictive-kernel", "0",
]

# BT = ship-v4 + flow-prior coasting, exit kill at the frame edge and colour
# from the track's stored colour, selected on the 71-case benchmark. The
# default preset.
BT_CONFIG = SHIP_CONFIG_V4 + [
    "--coast-predict", "flow", "--exit-kill", "1", "--color-source", "track",
]

# --preset name -> tracker flags. Later presets add a key here; explicit
# --extra flags are appended after the preset and win (argparse keeps the
# last value).
PRESETS = {
    "bt": BT_CONFIG,
    "ship-v4": SHIP_CONFIG_V4,
}
DEFAULT_PRESET = "bt"
# SHIP_CONFIG = the DEFAULT preset's flags (bt), kept for callers that
# import it. Code that must reproduce the published v4 numbers uses
# SHIP_CONFIG_V4 explicitly.
SHIP_CONFIG = PRESETS[DEFAULT_PRESET]


class UserError(ValueError):
    """A problem with the user's inputs: one plain-English line, exit code 2 (a ValueError for cbtv.track)."""


class _Parser(argparse.ArgumentParser):
    """argparse with one-line errors: ``[cbtv] error: ...`` and exit code 2 (no usage block)."""

    def error(self, message):
        self.exit(2, f"[cbtv] error: {message} (see cbtv --help)\n")


def version() -> str:
    try:
        from importlib.metadata import version as _v
        return _v("cbtv")
    except Exception:
        import re
        m = re.search(r'^version\s*=\s*"([^"]+)"',
                      (HERE / "pyproject.toml").read_text(encoding="utf-8"), re.M)
        return m.group(1) if m else "unknown"


def check_video(video: Path):
    if not video.exists():
        raise UserError(f"input video not found: {video}")
    import cv2
    cap = cv2.VideoCapture(str(video))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        raise UserError(f"cannot read video {video} (unsupported codec or not a "
                        f"video?); try converting it to mp4 (H.264) or avi")


def check_csv_detections(path, r_min: int, r_max: int, n_first: int = 5):
    """Cheap checks on a --detector csv file (no detector run)."""
    if not path:
        raise UserError("--detector csv needs --detections-csv FILE "
                        "(columns frame,x,y,r[,score])")
    p = Path(path)
    if not p.exists():
        raise UserError(f"detections CSV not found: {p}")
    try:
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            missing = [c for c in ("frame", "x", "y", "r") if c not in (rd.fieldnames or [])]
            if missing:
                raise UserError(f"detections CSV {p.name} is missing column(s) "
                                f"{', '.join(missing)}; need frame,x,y,r[,score]")
            first, radii = 0, []
            for line, row in enumerate(rd, start=2):
                if any(row[c] is None for c in ("frame", "x", "y", "r")):
                    raise UserError(f"detections CSV {p.name} line {line} has too few columns; "
                                    f"need frame,x,y,r[,score]")
                fr = int(float(row["frame"]))
                x, y, r = float(row["x"]), float(row["y"]), float(row["r"])
                if not all(math.isfinite(v) for v in (x, y, r)):
                    raise UserError(f"detections CSV {p.name} line {line}: x, y and r must be finite numbers "
                                    f"(got x={row['x']}, y={row['y']}, r={row['r']})")
                if r <= 0:
                    raise UserError(f"detections CSV {p.name} line {line}: radius r must be > 0 (got {row['r']})")
                if fr < n_first:
                    first += 1
                radii.append(r)
    except UserError:
        raise
    except (OSError, ValueError, OverflowError, csv.Error) as e:
        raise UserError(f"cannot parse detections CSV {p.name}: {e}")  # noqa: B904 -- user-facing message, chaining unchanged
    if first == 0:
        raise UserError(f"no detections in the first {n_first} processed frames of "
                        f"{p.name}; check the detector's radius range / threshold")
    radii.sort()
    med = radii[len(radii) // 2]
    if med < 0.7 * r_min or med > 1.3 * r_max:
        lo, hi = max(1, int(round(0.75 * med))), int(round(1.25 * med)) + 1
        raise UserError(f"median detection radius {med:.1f} px is far outside "
                        f"--r-min {r_min} / --r-max {r_max}; try --r-min {lo} --r-max {hi}")


def check_args(a):
    """Option checks that need no input file: ranges, detector name, output dir, --extra."""
    if a.frame_step < 1:
        raise UserError(f"--frame-step must be 1 or more (got {a.frame_step})")
    if a.max_frames != -1 and a.max_frames < 1:
        raise UserError(f"--max-frames must be -1 (all frames) or 1 or more (got {a.max_frames})")
    if a.r_min < 1 or a.r_max < a.r_min:
        raise UserError(f"need 1 <= --r-min <= --r-max (got --r-min {a.r_min} --r-max {a.r_max})")
    from cbtv.detectors import available
    if a.detector not in available():
        raise UserError(f"unknown --detector {a.detector!r}; choose from {', '.join(available())}")
    if a.detector == "ws_vgl_log":  # GPU-only strategy: say so up front instead of 0 detections
        from cbtv.detect import gpu_backend as gb
        if not (gb.HAS_CUPY and gb.HAS_CUPY_RAWKERNEL):
            raise UserError("--detector ws_vgl_log runs on the GPU only (CuPy + CUDA) and has no CPU path; "
                            "on CPU use --detector ws_multi_vote, csv or bayes_adapt")
    out = Path(a.output_dir)
    if out.exists() and not out.is_dir():
        raise UserError(f"--output-dir {out} exists and is not a directory")
    try:
        shlex.split(a.extra)
    except ValueError as e:
        raise UserError(f"cannot parse --extra {a.extra!r}: {e}") from None


def gpu_available() -> bool:
    """True if CuPy works on this machine, including compiling a CUDA kernel (see cbtv.gpu.cupy_works)."""
    try:
        from cbtv.gpu import cupy_works
        return cupy_works()
    except Exception:
        return False


def id_colour(tid: int):
    """Persistent, visually distinct BGR colour for a track id."""
    import colorsys
    h = (tid * 0.61803398875) % 1.0          # golden-ratio hue walk
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def is_coasting(row) -> bool:
    """A tracks.csv row with no detection this frame. `missed` is the
    consecutive-miss COUNT (0 = matched), not a flag."""
    try:
        return int(float(row.get("missed") or 0)) > 0
    except ValueError:
        return False


def annotate(video: Path, tracks_csv: Path, out_mp4: Path, frame_step: int,
             label: str):
    """Re-render the input video with persistent per-track colours + IDs."""
    import cv2
    by_frame = defaultdict(list)
    with open(tracks_csv, newline="") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])].append(
                (int(row["track_id"]), float(row["x"]), float(row["y"]),
                 float(row["r"]), is_coasting(row)))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter.fourcc(*"mp4v"),
                         max(1.0, fps / frame_step), (W, H))
    src_idx, proc_idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx % frame_step == 0:
            for (tid, x, y, r, missed) in by_frame.get(proc_idx, []):
                col = (150, 150, 150) if missed else id_colour(tid)
                thick = 1 if missed else 2
                cv2.circle(frame, (int(round(x)), int(round(y))),
                           int(max(3, round(r))), col, thick, cv2.LINE_AA)
                if label == "id" and not missed:
                    cv2.putText(frame, str(tid),
                                (int(x) + int(r) + 2, int(y) - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                                cv2.LINE_AA)
            wr.write(frame)
            proc_idx += 1
            if proc_idx > max(by_frame, default=0):
                break
        src_idx += 1
    cap.release()
    wr.release()


def summarise(tracks_csv: Path, out_txt: Path):
    frames_of = defaultdict(list)
    per_frame: defaultdict = defaultdict(int)
    coasting = 0
    n_rows = 0
    states: defaultdict = defaultdict(int)  # --row-tags 1 only
    with open(tracks_csv, newline="") as f:
        for row in csv.DictReader(f):
            fr, tid = int(row["frame"]), int(row["track_id"])
            frames_of[tid].append(fr)
            n_rows += 1
            if row.get("state"):
                states[row["state"]] += 1
            if is_coasting(row):
                coasting += 1
            else:
                per_frame[fr] += 1
    if not frames_of:
        out_txt.write_text("No tracks produced.\n")
        return
    n_frames = max(max(v) for v in frames_of.values()) + 1
    lengths = sorted(len(set(v)) for v in frames_of.values())
    full = sum(1 for v in frames_of.values()
               if min(v) == 0 and max(v) == n_frames - 1)
    med = lengths[len(lengths) // 2]
    dets = sorted(per_frame.values())
    lines = [
        "CBTV run summary",
        "================",
        f"frames processed          : {n_frames}",
        f"unique tracks             : {len(frames_of)}",
        f"track length median/mean  : {med} / {sum(lengths)/len(lengths):.1f} frames",
        f"tracks spanning full video: {full} "
        f"({100.0*full/len(frames_of):.1f}%)",
        f"particles per frame median: {dets[len(dets)//2] if dets else 0}",
        f"coasting rows (missed>0)  : {coasting} "
        f"({100.0*coasting/max(1,n_rows):.1f}% of rows)",
        "",
        "Reading this: on a well-tracked video 'tracks spanning full video'",
        "should approach the true particle count and 'unique tracks' should",
        "not be much larger than it (excess = fragmentation). A high",
        "coasting percentage means detection is struggling (lighting,",
        "focus, radius range) - see the tuning section of docs/user_guide.md.",
    ]
    if states:
        lines += ["", "row states (--row-tags):"]
        lines += [f"  {k:<20}: {v} ({100.0*v/max(1,n_rows):.1f}%)"
                  for k, v in sorted(states.items())]
    meta = tracks_csv.with_name(tracks_csv.stem + "_meta.csv")
    if meta.exists():
        births: defaultdict = defaultdict(int)
        deaths: defaultdict = defaultdict(int)
        with open(meta, newline="") as f:
            for row in csv.DictReader(f):
                births[row["birth_reason"]] += 1
                deaths[row["death_reason"]] += 1
        lines += ["", f"track births ({meta.name}): " +
                  ", ".join(f"{k} {v}" for k, v in sorted(births.items())),
                  "track deaths: " +
                  ", ".join(f"{k} {v}" for k, v in sorted(deaths.items()))]
    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def build_parser(parser_cls=None) -> argparse.ArgumentParser:
    """The ``cbtv`` command line; :func:`cbtv.track` parses its keyword arguments with the same parser."""
    parser_cls = parser_cls or _Parser
    ap = parser_cls(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="experimental video file")
    ap.add_argument("--input", default=None, help="same as the positional video")
    ap.add_argument("--version", action="version", version=f"cbtv {version()}")
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS),
                    help="tracking configuration [bt]; ship-v4 = the published v4 "
                         "configuration (reproduces the paper's v4 numbers)")
    ap.add_argument("--output-dir", default="results",
                    help="output directory [results]")
    ap.add_argument("--detector", default=None,
                    help="detection preset [ws_multi_vote, or csv when "
                         "--detections-csv is given]; others: "
                         "bayes_adapt, ws_vgl_log, log, cellpose (needs pip "
                         "install cellpose), csv (bring your own detections)")
    ap.add_argument("--detections-csv", default=None,
                    help="[--detector csv] your own frame,x,y,r,score CSV")
    ap.add_argument("--gpu", default="auto", choices=["auto", "1", "0"],
                    help="auto = use CUDA via CuPy if available [auto]")
    ap.add_argument("--gpu-device", type=int, default=0,
                    help="CUDA device index [0]")
    ap.add_argument("--r-min", type=int, default=6,
                    help="minimum particle radius, px [6]")
    ap.add_argument("--r-max", type=int, default=10,
                    help="maximum particle radius, px [10]")
    ap.add_argument("--input-polarity", default="auto",
                    choices=["auto", "dark", "bright"],
                    help="background polarity: auto detects it from the "
                         "first frame [auto]; dark = particles brighter "
                         "than the background (no preprocessing); bright = "
                         "particles on a lit background (mattes it to black)")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="process every Nth frame [1]")
    ap.add_argument("--max-frames", type=int, default=-1,
                    help="stop after N frames (-1 = all) [-1]")
    ap.add_argument("--label", default="id", choices=["id", "none"],
                    help="draw track IDs next to circles [id]")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed; same seed + inputs = same tracks [0]")
    ap.add_argument("--verbose", type=int, default=0,
                    help="per-frame tracker diagnostics 0|1 [0]")
    ap.add_argument("--extra", default="",
                    help='extra raw cbtv.run_tracker flags, quoted, e.g. '
                         '--extra "--colour-ema 0.5"')
    return ap


def resolve(a) -> None:
    """Resolve the detector (``csv`` when only ``--detections-csv`` is given) and check the options.

    Raises :class:`UserError` on a bad combination; shared by :func:`main` and :func:`cbtv.track`.
    """
    if a.detector is None:
        a.detector = "csv" if a.detections_csv else "ws_multi_vote"
    elif a.detector != "csv" and a.detections_csv:
        raise UserError(f"--detections-csv only works with --detector csv, but "
                        f"--detector {a.detector} was given; drop one of the two flags")
    check_args(a)


def main_cptv_alias() -> int:
    """The former ``cptv`` command name; prints one deprecation line, then runs :func:`main`."""
    print("[cbtv] note: the 'cptv' command is deprecated; use 'cbtv' (same options)", file=sys.stderr)
    return main()


def main() -> int:
    """Entry point of the ``cbtv`` command (and ``python run_cbtv.py``).

    Parses the command line, resolves the detector (``csv`` when only
    ``--detections-csv`` is given) and runs :func:`run`. Input problems raise
    :class:`UserError` and are reported as one ``[cbtv] error:`` line with
    exit code 2 (full traceback with ``--verbose 1``).

    Returns the process exit code: 0 on success, 2 on an input error.
    """
    ap = build_parser()
    a = ap.parse_args()
    if not (a.video or a.input):
        ap.error("give the input video, e.g. cbtv my_experiment.mp4")
    try:
        resolve(a)
        return run(a)
    except UserError as e:
        if a.verbose:
            raise
        print(f"[cbtv] error: {e}", file=sys.stderr)
        return 2


def run(a) -> int:
    """Check the inputs, run the tracker core, then render and summarise.

    ``a`` is the parsed namespace from :func:`main`. Checks run before any
    tracking (video readable, CuPy for ``--gpu 1``, the detections CSV for
    ``--detector csv``). The tracker runs as ``python -m cbtv.run_tracker``
    in the caller's directory with the chosen preset's flags, then
    ``annotated.mp4`` and ``summary.txt`` are written next to its
    ``tracks.csv`` and ``overlay.mp4`` in ``--output-dir``.

    Returns 0 on success; exits with a message if the tracker fails.
    """
    video = Path(a.video or a.input)
    check_video(video)
    if a.gpu == "1" and not gpu_available():
        raise UserError('--gpu 1 but no working CuPy/CUDA was found; install it with '
                        'pip install "cbtv[gpu]" or run on CPU with --gpu 0')
    if a.detector == "csv":
        check_csv_detections(a.detections_csv, a.r_min, a.r_max)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    use_gpu = a.gpu == "1" or (a.gpu == "auto" and gpu_available())
    if a.gpu == "auto" and not use_gpu:
        print("[WARN] --gpu auto: no working CuPy/CUDA found, running on CPU (slower).")
    print(f"[cbtv] device: {'GPU (CuPy/CUDA)' if use_gpu else 'CPU'}"
          + ("" if a.gpu != "auto" else "  (auto-detected)"))

    tracks_csv = out / "tracks.csv"
    argv = [sys.executable, "-u", "-m", "cbtv.run_tracker",
            "--input", str(video),
            "--output-video", str(out / "overlay.mp4"),
            "--output-csv", str(tracks_csv),
            "--detector", a.detector,
            "--use-gpu", "1" if use_gpu else "0",
            "--gpu-device", str(a.gpu_device),
            "--r-min", str(a.r_min), "--r-max", str(a.r_max),
            "--frame-step", str(a.frame_step),
            "--max-frames", str(a.max_frames),
            "--input-polarity", a.input_polarity,
            # first-frame debug images (debug_hybrid/) only on request; tracks do not depend on it
            "--dump-first-frame-corr", "1" if a.verbose else "0",
            "--dump-dir", str(out / "debug_hybrid"),
            "--seed", str(a.seed),
            "--verbose", str(a.verbose)] + PRESETS[a.preset]
    if a.detections_csv:
        argv += ["--detections-csv", a.detections_csv]
    if a.extra:
        argv += shlex.split(a.extra)
    if a.verbose:
        print("[cbtv] tracker argv:", " ".join(argv[1:]))

    print("[cbtv] tracking", video.name, "...")
    # Run in the caller's directory so relative --input/--output-dir/--extra
    # paths resolve as typed (used to be cwd=HERE, which broke them unless
    # run from the package folder, e.g. after pip install). HERE goes on
    # PYTHONPATH so an uninstalled checkout still finds cbtv/.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(HERE), env.get("PYTHONPATH", "")) if p)
    rc = subprocess.run(argv, env=env).returncode
    if rc != 0 or not tracks_csv.exists():
        sys.exit(f"[cbtv] tracker failed (rc={rc}) - rerun with --verbose 1")

    if getattr(a, "render", True):  # always on the CLI; cbtv.track(render=False) skips it
        print("[cbtv] rendering annotated video ...")
        annotate(video, tracks_csv, out / "annotated.mp4", a.frame_step, a.label)
    summarise(tracks_csv, out / "summary.txt")
    print(f"[cbtv] done - outputs in {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
