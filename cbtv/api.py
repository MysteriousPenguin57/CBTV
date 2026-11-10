"""Public Python API: ``cbtv.track(...)`` and ``cbtv.presets()``.

``track`` is the ``cbtv`` command as a function: its keyword arguments are turned into the command's options, parsed
by the same parser (:func:`run_cbtv.build_parser`), checked by the same code (:func:`run_cbtv.resolve`) and run by
the same :func:`run_cbtv.run` (the tracker core as ``python -m cbtv.run_tracker`` with the preset's flags). There is
no second implementation, so a ``track`` call and the equivalent ``cbtv`` command line give byte-identical
``tracks.csv``.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# keyword arguments of track() that map 1:1 onto `cbtv` options (underscores -> dashes)
_OPTIONS = ("detector", "gpu_device", "r_min", "r_max", "input_polarity", "frame_step", "max_frames", "label",
            "seed", "verbose", "extra")
_DET_COLUMNS = ("frame", "x", "y", "r")


class _ApiParser(argparse.ArgumentParser):
    """The cbtv parser, raising ValueError instead of exiting the interpreter."""

    def error(self, message):
        raise ValueError(f"cbtv.track: {message}")


def _run_cbtv():
    import run_cbtv  # top-level module of the cbtv distribution (the `cbtv` command)
    return run_cbtv


def presets() -> dict[str, list[str]]:
    """The tracking presets and the tracker flags each one adds.

    Returns:
        ``{name: flags}``, e.g. ``{"bt": [...], "ship-v4": [...]}``; ``"bt"`` is the default. The lists are
        copies of ``run_cbtv.PRESETS``.
    """
    return {name: list(flags) for name, flags in _run_cbtv().PRESETS.items()}


def _gpu_flag(gpu: Any) -> str:
    if gpu in ("auto", None):
        return "auto"
    if gpu is True or gpu in (1, "1", "on", "gpu"):
        return "1"
    if gpu is False or gpu in (0, "0", "off", "cpu"):
        return "0"
    raise ValueError(f"cbtv.track: gpu must be 'auto', True or False (got {gpu!r})")


def track(video: str | os.PathLike | None = None, detections: Any = None, preset: str = "bt",
          gpu: Any = "auto", **overrides: Any) -> pd.DataFrame:
    """Track the particles in a video and return the tracks as a DataFrame.

    Runs exactly what the ``cbtv`` command runs (same option parsing, checks, tracker flags and preset), so the
    result equals the ``tracks.csv`` that ``cbtv VIDEO --preset PRESET ...`` writes.

    Args:
        video: Path of the input video (required).
        detections: Your own detections: a path to a ``frame,x,y,r[,score]`` CSV, or a DataFrame with those
            columns (written to a temporary CSV). Given -> the ``csv`` detector, as ``--detections-csv`` does.
        preset: Tracking configuration, a key of :func:`presets`: ``"bt"`` (default) or ``"ship-v4"`` (the
            published v4 configuration).
        gpu: ``"auto"`` (CuPy/CUDA if available, else CPU), ``True`` (GPU, error if unavailable) or ``False``
            (CPU).
        **overrides: Other ``cbtv`` options with underscores: ``detector``, ``r_min``, ``r_max``
            (particle radius range in px), ``gpu_device``, ``input_polarity``, ``frame_step``, ``max_frames``,
            ``label``, ``seed``, ``verbose``, ``extra`` (raw tracker flags, one string). Also ``output_dir``:
            keep all outputs (``tracks.csv``, ``overlay.mp4``, ``summary.txt``, ...) there instead of in a
            temporary directory, and ``render`` (default False): also write ``annotated.mp4``, as the command
            does.

    Returns:
        The rows of ``tracks.csv`` (columns ``frame, track_id, x, y, r, missed``), one row per track per
        processed frame. ``df.attrs["output_dir"]`` is the output directory when ``output_dir`` was given.

    Raises:
        ValueError: Bad inputs (missing or unreadable video, bad detections, unknown option or preset); the
            message is the one the command prints after ``[cbtv] error:``.
        RuntimeError: The tracker core failed.
        ImportError: pandas is not installed (``pip install "cbtv[api]"``).

    Example:
        >>> import cbtv
        >>> df = cbtv.track("clip.mp4", detections="detections.csv", gpu=False)
        >>> df.groupby("track_id").size().describe()
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('cbtv.track needs pandas: pip install "cbtv[api]"') from e
    if video is None:
        raise ValueError("cbtv.track: video is required (the path of the input video)")
    output_dir = overrides.pop("output_dir", None)
    render = bool(overrides.pop("render", False))
    unknown = sorted(set(overrides) - set(_OPTIONS))
    if unknown:
        raise TypeError(f"cbtv.track: unknown option(s) {', '.join(unknown)}; known: {', '.join(_OPTIONS)}, "
                        "output_dir, render")
    rc = _run_cbtv()
    with tempfile.TemporaryDirectory(prefix="cbtv_track_") as tmp:
        out = Path(output_dir) if output_dir is not None else Path(tmp) / "out"
        argv = [os.fspath(video), "--preset", preset, "--gpu", _gpu_flag(gpu), "--output-dir", str(out)]
        if detections is not None:
            if isinstance(detections, pd.DataFrame):
                missing = [c for c in _DET_COLUMNS if c not in detections.columns]
                if missing:
                    raise ValueError(f"cbtv.track: detections DataFrame lacks column(s) {', '.join(missing)}; "
                                     "need frame,x,y,r[,score]")
                det_csv = Path(tmp) / "detections.csv"
                detections.to_csv(det_csv, index=False)
                detections = det_csv
            argv += ["--detections-csv", os.fspath(detections)]
        for key, value in overrides.items():
            if value is not None:
                argv.append(f"--{key.replace('_', '-')}={value}")  # '=' form: --extra values start with '--'
        a = rc.build_parser(_ApiParser).parse_args(argv)
        a.render = render
        rc.resolve(a)
        try:
            rc.run(a)
        except SystemExit as e:  # run() exits with a message when the tracker core fails
            raise RuntimeError(str(e.code)) from None
        df = pd.read_csv(out / "tracks.csv")
    if output_dir is not None:
        df.attrs["output_dir"] = str(out)
    return df
