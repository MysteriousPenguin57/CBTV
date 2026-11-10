"""C018: the public Python API cbtv.track / cbtv.presets runs the cbtv command's code path."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OPENCV_OPENCL_RUNTIME": "disabled", "PYTHONHASHSEED": "0"}


@pytest.fixture
def cpu_env(monkeypatch):
    for k, v in CPU_ENV.items():
        monkeypatch.setenv(k, v)


def test_presets_match_the_cli():
    import cbtv
    import run_cbtv
    assert cbtv.presets() == run_cbtv.PRESETS
    assert set(cbtv.presets()) == {"bt", "ship-v4"}
    cbtv.presets()["bt"].append("--mutated")  # a copy, not the CLI's table
    assert "--mutated" not in run_cbtv.PRESETS["bt"]


def test_import_cbtv_does_not_import_pandas():
    code = "import sys; sys.path.insert(0, r'%s'); import cbtv; print('pandas' in sys.modules)" % ROOT
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


@pytest.mark.parametrize("kwargs, exc, needle", [
    ({}, ValueError, "video is required"),
    ({"video": "no_such_video.mp4", "gpu": False}, ValueError, "input video not found"),
    ({"video": "x.mp4", "bogus": 1}, TypeError, "unknown option(s) bogus"),
    ({"video": "x.mp4", "gpu": "maybe"}, ValueError, "gpu must be"),
    ({"video": str(FIX / "synth_io" / "clip.avi"), "preset": "nope", "gpu": False}, ValueError, "invalid choice"),
    ({"video": str(FIX / "synth_io" / "clip.avi"), "detections": pd.DataFrame({"frame": [0], "x": [1.0]}),
      "gpu": False}, ValueError, "lacks column(s) y, r"),
    ({"video": str(FIX / "synth_io" / "clip.avi"), "r_min": 9, "r_max": 3, "gpu": False}, ValueError, "--r-min"),
])
def test_bad_inputs_raise(kwargs, exc, needle, cpu_env):
    import cbtv
    with pytest.raises(exc) as ei:
        cbtv.track(**kwargs)
    assert needle in str(ei.value)


def test_track_equals_the_cli_run(tmp_path, cpu_env):
    """Same options -> byte-identical tracks.csv from cbtv.track and from the cbtv command (short synth_io run)."""
    import cbtv
    video, det = FIX / "synth_io" / "clip.avi", FIX / "synth_io" / "detections.csv"
    cli_out = tmp_path / "cli"
    subprocess.run([sys.executable, str(ROOT / "run_cbtv.py"), str(video), "--detections-csv", str(det),
                    "--gpu", "0", "--max-frames", "8", "--output-dir", str(cli_out)],
                   env=dict(os.environ, **CPU_ENV), capture_output=True, check=True)
    api_out = tmp_path / "api"
    df = cbtv.track(video, detections=det, gpu=False, max_frames=8, output_dir=api_out)
    assert (api_out / "tracks.csv").read_bytes() == (cli_out / "tracks.csv").read_bytes()
    assert df.equals(pd.read_csv(cli_out / "tracks.csv"))
    assert list(df.columns) == ["frame", "track_id", "x", "y", "r", "missed"]
    assert df.attrs["output_dir"] == str(api_out)
    assert not (api_out / "annotated.mp4").exists()  # render=False by default
    # detections as a DataFrame and an --extra pass-through give the same result
    df2 = cbtv.track(str(video), detections=pd.read_csv(det), gpu=False, max_frames=8, extra="--seed 0")
    assert df2.equals(df)
    assert "output_dir" not in df2.attrs


@pytest.mark.slow
@pytest.mark.parametrize("preset, golden", [("bt", "golden_gpu_tracks_bt.csv"),
                                            ("ship-v4", "golden_gpu_tracks.csv")])
def test_primary60_dataframe_equals_golden(preset, golden, tmp_path, cpu_env):
    """C018 acceptance: on primary60 (CPU) the DataFrame equals the committed golden of the preset."""
    import cbtv
    fx = FIX / "primary60"
    df = cbtv.track(fx / "clip.mp4", detections=fx / "detections.csv", preset=preset, gpu=False,
                    output_dir=tmp_path)
    assert (tmp_path / "tracks.csv").read_bytes() == (fx / golden).read_bytes()
    assert df.equals(pd.read_csv(fx / golden))
