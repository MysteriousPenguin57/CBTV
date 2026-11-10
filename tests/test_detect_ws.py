"""C005: ws_multi_vote / ws_vgl_log detections on primary60 frames 0-4 equal the pre-de-vendoring output.

The fixture tests/fixtures/detect_ws/primary60_f0-4.csv was written by tools/make_detect_ws_fixture.py on main
before cbtv/detect/ existed (top-level detection_benchmark / strategy_combination_benchmark). CPU path."""
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))




def test_ws_detections_unchanged():
    # Run the detectors in a fresh process with CUDA hidden: conftest may already have loaded CuPy here, and
    # with CuPy the *_gpu strategies take the GPU path, while the fixture is the CPU path.
    import subprocess
    code = ("import csv, sys; sys.path.insert(0, r'%s'); import make_detect_ws_fixture as mk; "
            "csv.writer(sys.stdout).writerows(mk.detect_all())" % (ROOT / 'tools'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True).stdout
    after = [tuple(r) for r in csv.reader(out.splitlines()) if r]
    import make_detect_ws_fixture as mk
    with open(mk.OUT, newline='') as fh:
        rd = csv.reader(fh)
        assert next(rd) == ['strategy', 'frame', 'x', 'y', 'r', 'score']
        before = [tuple(r) for r in rd]
    assert len(after) == len(before)
    assert after == before
