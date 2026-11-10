"""C015 slow tests: CBTV itself in the loop (CPU runs on lossless decimated / reversed synth_io clips, ~1 min)."""
import json
import subprocess
import sys

import pytest
from conftest import FIX, ROOT

from tools.realdata import consensus, score, validate
from tools.realdata.tracksio import read_detections, read_gt

pytestmark = pytest.mark.slow
PY = sys.executable
SIO = FIX / "synth_io"


def cli(*args):
    r = subprocess.run([PY, "-m", "tools.realdata", *map(str, args)], cwd=str(ROOT), capture_output=True, text=True)
    return r


def test_cbtv_decimated_cells_scored_against_cbtv_free_reference(tmp_path):
    d = read_detections(SIO / "detections.csv")
    gt = read_gt(SIO / "gt.csv")
    ms = consensus.builtin_members(d, ("relax", "nn", "lap"), k=2)
    ref = consensus.write(consensus.build(d, ms, k=2), d, tmp_path / "ref")
    rt, _, meta = consensus.load(ref)
    runs, files = validate.cbtv_cell(d, 2, SIO / "clip.avi", tmp_path)
    assert len(runs) == 2 and all(len(t) for t, _ in runs)
    consensus.check_scoring(meta, "cbtv", 2, files)            # CBTV-free reference: allowed
    sp = score.score_offsets(runs, rt, 8.0)
    st = score.score_offsets(runs, gt, 8.0)
    assert sp["klinks"] > 2000 and st["klinks"] > sp["klinks"]
    assert abs(sp["id_err"] - st["id_err"]) < 0.01


def test_cli_pipeline_with_cbtv_member_and_refusal(tmp_path):
    dets = SIO / "detections.csv"
    r = cli("decimate", "--detections", dets, "--k", 2, "--out", tmp_path / "dec", "--video", SIO / "clip.avi")
    assert r.returncode == 0, r.stderr
    r = cli("reverse", "--detections", dets, "--out", tmp_path / "rev.csv", "--video", SIO / "clip.avi")
    assert r.returncode == 0, r.stderr
    runs = {}
    for name, clip, seq in [("o0", tmp_path / "dec" / "dets_k2_o0.avi", tmp_path / "dec" / "dets_k2_o0.csv"),
                            ("o1", tmp_path / "dec" / "dets_k2_o1.avi", tmp_path / "dec" / "dets_k2_o1.csv"),
                            ("rev", tmp_path / "rev.avi", tmp_path / "rev.csv")]:
        runs[name] = (validate.run_cbtv(clip, seq, tmp_path / ("run_" + name)), seq)
    fwd = SIO / "cpu_baseline_tracks.csv"
    r = cli("fb", "--detections", dets, "--forward", fwd, "--backward", "%s@%s" % runs["rev"])
    assert r.returncode == 0, r.stderr
    rate = json.loads(r.stdout)["fb_rate"]
    assert 0.9 < rate <= 1.0
    args = ["consensus", "--detections", dets, "--out", tmp_path / "ref_cbtv", "--k", 2, "--builtin", "nn,lap",
            "--member", "cbtv=%s" % fwd, "--member-rev", "cbtv=%s@%s" % runs["rev"],
            "--member-dec", "cbtv=%s@%s" % runs["o0"], "--member-dec", "cbtv=%s@%s" % runs["o1"]]
    r = cli(*args)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["cbtv_derived"] is True and out["reference_links"] > 5000
    meta = json.loads((tmp_path / "ref_cbtv" / "reference.meta.json").read_text())
    cm = [m for m in meta["members"] if m["name"] == "cbtv"][0]
    assert cm["forward_backward"] and cm["steps"] == [1, 2] and len(cm["sources"]) == 4
    # CBTV at step 2 against this reference: refused, whatever name it is given
    for tracker in ("cbtv", "anonymous"):
        r = cli("score", "--reference", tmp_path / "ref_cbtv", "--tracker", tracker,
                "--tracks", "%s@%s" % runs["o0"], "--tracks", "%s@%s" % runs["o1"])
        assert r.returncode == 3, (tracker, r.stdout, r.stderr)
    # and accepted against the CBTV-free reference
    r = cli("consensus", "--detections", dets, "--out", tmp_path / "ref_free", "--k", 2, "--builtin", "relax,nn,lap")
    assert r.returncode == 0, r.stderr
    r = cli("score", "--reference", tmp_path / "ref_free", "--tracker", "cbtv",
            "--tracks", "%s@%s" % runs["o0"], "--tracks", "%s@%s" % runs["o1"])
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res["step"] == 2 and res["klinks"] > 2000
