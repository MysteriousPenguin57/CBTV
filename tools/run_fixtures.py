"""Run the SHIP argv on each fixture. Usage: run_fixtures.py gpu|cpu <outdir> [fixture ...] [--bare-env] [--preset P]
--preset ship-v4 (default; the flags below, which produced golden_gpu_tracks.csv) or bt (ship-v4 + the 3 BT
flags, for golden_gpu_tracks_bt.csv).
gpu: --use-gpu 1 --gpu-device 0 ; cpu: --use-gpu 0 with CUDA hidden, OpenCL off, PYTHONHASHSEED=0.
--bare-env (cpu): only hide CUDA; no OPENCV_OPENCL_RUNTIME / PYTHONHASHSEED (the tracker itself
turns OpenCL off on --use-gpu 0 and seeds its RNG).
Writes <outdir>/<id>/tracks.csv, run.log (argv on line 1), wall.txt. Deletes tracks.mp4."""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FIX = HERE / "tests" / "fixtures"
SHIP = ["--use-constellations", "1", "--cluster-rounds", "2", "--cluster-max-per-frame", "256",
        "--cluster-cap-auto", "1", "--cluster-cap-alpha", "0.324", "--cluster-cap-max", "2048",
        "--mu-source", "piv", "--piv-max-disp-ratio", "1.0", "--match-op", "graph-struct",
        "--match-gate-d", "1.1", "--w-kernel", "0.1", "--w-dist", "0.8", "--w-color", "0.1",
        "--use-affine-field", "0", "--opt-predictive-kernel", "0"]
BT = SHIP + ["--coast-predict", "flow", "--exit-kill", "1", "--color-source", "track"]
CLIP = {"primary60": "clip.mp4", "m05_30": "clip.mp4", "synth_io": "clip.avi"}
args = sys.argv[1:]
preset = "ship-v4"
if "--preset" in args:
    i = args.index("--preset"); preset = args[i + 1]; del args[i:i + 2]
FLAGS = {"ship-v4": SHIP, "bt": BT}[preset]
argv = [a for a in args if a != "--bare-env"]
bare = "--bare-env" in args
mode, out = argv[0], Path(argv[1])
ids = argv[2:] or list(CLIP)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
if mode == "cpu":
    env.update(CUDA_VISIBLE_DEVICES="-1")
    if bare:
        for k in ("OPENCV_OPENCL_RUNTIME", "PYTHONHASHSEED"):
            env.pop(k, None)
    else:
        env.update(OPENCV_OPENCL_RUNTIME="disabled", PYTHONHASHSEED="0")
for fid in ids:
    d = out / fid; d.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-u", "-m", "cbtv.run_tracker", "--input", str(FIX / fid / CLIP[fid]),
            "--output-video", str(d / "tracks.mp4"), "--output-csv", str(d / "tracks.csv"),
            "--frame-step", "1", "--detector", "csv", "--detections-csv", str(FIX / fid / "detections.csv"),
            "--use-gpu", "1" if mode == "gpu" else "0", "--gpu-device", "0",
            "--r-min", "6", "--r-max", "10"] + FLAGS
    t0 = time.time()
    with open(d / "run.log", "w", encoding="utf-8") as lg:
        lg.write(" ".join(argv) + "\n"); lg.flush()
        rc = subprocess.call(argv, cwd=str(HERE), stdout=lg, stderr=subprocess.STDOUT, env=env)
    (d / "wall.txt").write_text("%.1f\n" % (time.time() - t0))
    try: (d / "tracks.mp4").unlink()
    except OSError: pass
    print(fid, mode, "rc", rc, "wall %.1f" % (time.time() - t0), flush=True)
