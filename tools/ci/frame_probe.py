"""C014 diagnosis: which OpenCV video backend decodes each fixture clip, and hashes of the first decoded frames.
Run on two OSes and compare: equal hashes mean decoding is bit-identical, so any track difference is downstream."""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLIPS = {"primary60": "clip.mp4", "m05_30": "clip.mp4", "synth_io": "clip.avi"}


def probe(path, api=cv2.CAP_ANY, n=3):
    cap = cv2.VideoCapture(str(path), api)
    if not cap.isOpened():
        return "not opened", []
    name = cap.getBackendName()
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append("%s mean=%.4f" % (hashlib.sha1(np.ascontiguousarray(f).tobytes()).hexdigest()[:12], f.mean()))
    cap.release()
    return name, out


print("opencv", cv2.__version__, "numpy", np.__version__, sys.platform)
for fx, clip in CLIPS.items():
    p = ROOT / "tests" / "fixtures" / fx / clip
    for label, api in (("default", cv2.CAP_ANY), ("ffmpeg", cv2.CAP_FFMPEG)):
        name, frames = probe(p, api)
        print("%-9s %-7s backend=%-12s %s" % (fx, label, name, " | ".join(frames)))
