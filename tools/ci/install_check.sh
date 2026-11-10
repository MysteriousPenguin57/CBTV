#!/usr/bin/env bash
# C001: Linux clean-install check for one Python interpreter (CPU only).
# Usage: tools/ci/install_check.sh [python-exe] [workdir]
#   python-exe  default python3; a fresh venv is created from it in <workdir>/venv
#   workdir     default a mktemp dir; logs + smoke outputs land there
# Env: SKIP_VENV=1 uses the given python directly (CI already has a clean one).
# Exit code = number of failed steps. Run from anywhere; paths resolve to the repo root.
set -u
PY=${1:-python3}
WORK=${2:-$(mktemp -d)}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$WORK"
export CUDA_VISIBLE_DEVICES=-1 OPENCV_OPENCL_RUNTIME=disabled PYTHONHASHSEED=0 MPLBACKEND=Agg
FAILS=0
step() {  # step <name> <cmd...>: run, log to $WORK/<name>.log, print PASS/FAIL + first 30 lines on failure
  local name=$1; shift
  if "$@" >"$WORK/$name.log" 2>&1; then echo "PASS $name"
  else echo "FAIL $name (rc $?)"; head -30 "$WORK/$name.log" | sed 's/^/    /'; FAILS=$((FAILS+1)); fi
}

if [ "${SKIP_VENV:-0}" = 1 ]; then VPY=$PY
else
  "$PY" -m venv "$WORK/venv" || { echo "FAIL venv"; exit 99; }
  VPY=$WORK/venv/bin/python
fi
echo "python: $("$VPY" --version 2>&1)  workdir: $WORK"
step install "$VPY" -m pip install -q -r "$REPO/requirements.txt"
"$VPY" -m pip freeze >"$WORK/freeze.txt" 2>&1

cd "$REPO"
step import     "$VPY" -c "import cbtv, cbtv.run_tracker"
step help_mod   "$VPY" -m cbtv.run_tracker --help
step help_run   "$VPY" run_cbtv.py --help
step compileall "$VPY" -m compileall -q cbtv $(ls *.py)

# CPU smoke: 20 frames 256x256 (frames < 256 px trigger small-image mode, which rescales
# r_min/r_max to 2/4 and drops r=7 discs; cbtv/run_tracker.py:213), 10 coloured discs (r=7) on black, each moving 1 px/frame,
# plus the matching frame,x,y,r,score detections CSV (cbtv/csv_detector.py schema).
cat >"$WORK/make_smoke.py" <<'PYEOF'
import csv, sys
import cv2, numpy as np
out = sys.argv[1]
W = H = 256; N = 10; F = 20; R = 7
cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128)]
# 5 x 2 grid, 40 px apart; alternate rows move +x / +y so no two discs collide
start = [(40 + 40 * (i % 5), 60 + 100 * (i // 5)) for i in range(N)]
vel = [(1, 0) if i // 5 == 0 else (0, 1) for i in range(N)]
vw = cv2.VideoWriter(out + "/clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
assert vw.isOpened(), "VideoWriter(mp4v) failed"
with open(out + "/dets.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "x", "y", "r", "score"])
    for t in range(F):
        img = np.zeros((H, W, 3), np.uint8)
        for i in range(N):
            x, y = start[i][0] + vel[i][0] * t, start[i][1] + vel[i][1] * t
            cv2.circle(img, (x, y), R, cols[i], -1, cv2.LINE_AA)
            w.writerow([t, x, y, R, 1.0])
        vw.write(img)
vw.release()
PYEOF
cat >"$WORK/check_smoke.py" <<'PYEOF'
import csv, sys
from collections import Counter
rows = list(csv.DictReader(open(sys.argv[1])))
per_frame = Counter(int(float(r["frame"])) for r in rows)
ids = {r["track_id"] for r in rows}
live = Counter(int(float(r["frame"])) for r in rows if int(float(r.get("missed") or 0)) == 0)
good = sum(1 for t in range(20) if live.get(t, 0) == 10)
print("rows", len(rows), "track_ids", len(ids), "frames with 10 live tracks", good, "/ 20")
sys.exit(0 if len(ids) <= 12 and good >= 15 else 1)
PYEOF
step smoke_make "$VPY" "$WORK/make_smoke.py" "$WORK"
step smoke_track "$VPY" -m cbtv.run_tracker --input "$WORK/clip.mp4" --output-video "$WORK/out.mp4" \
  --output-csv "$WORK/tracks.csv" --detector csv --detections-csv "$WORK/dets.csv" --use-gpu 0
step smoke_check "$VPY" "$WORK/check_smoke.py" "$WORK/tracks.csv"
cat "$WORK/smoke_check.log" 2>/dev/null | sed 's/^/    /'
step smoke_run_cbtv "$VPY" run_cbtv.py "$WORK/clip.mp4" --output-dir "$WORK/run_cbtv_out" \
  --detector csv --detections-csv "$WORK/dets.csv" --gpu 0
echo "failed steps: $FAILS"
exit $FAILS
