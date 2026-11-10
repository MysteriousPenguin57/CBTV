"""Write tests/fixtures/detect_ws/primary60_f0-4.csv: ws_multi_vote / ws_vgl_log detections on primary60 frames 0-4
(C005 before/after check; floats in repr so the comparison is exact). Usage: python tools/make_detect_ws_fixture.py"""
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cbtv import ws_detector  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "detect_ws" / "primary60_f0-4.csv"


def detect_all():
    from cbtv.detect import gpu_backend as gb
    rows = []
    # ws_vgl_log_gpu is GPU-only and raises without CuPy; on CPU it used to contribute 0 rows
    gpu = gb.HAS_CUPY and gb.HAS_CUPY_RAWKERNEL
    strats = ("ws_multi_vote_gpu", "ws_vgl_log_gpu") if gpu else ("ws_multi_vote_gpu",)
    for strat in strats:
        ws_detector._STRATEGY_FN = None
        cap = cv2.VideoCapture(str(ROOT / "tests" / "fixtures" / "primary60" / "clip.mp4"))
        for f in range(5):
            ok, frame = cap.read()
            assert ok
            dets = ws_detector.detect_ws(frame, 6, 10, strat)[0]
            rows += [(strat, f) + tuple(repr(v) for v in d) for d in dets]
    return rows


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "frame", "x", "y", "r", "score"])
        w.writerows(detect_all())
    print(OUT)
