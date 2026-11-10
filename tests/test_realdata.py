"""C015 tools/realdata unit tests (fast; synthetic data generated here, fixtures read-only)."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import FIX, ROOT

from tools.realdata import audit, consensus, decimate, fb, linkers, score, synth, validate
from tools.realdata.tracksio import (
    prepare_detections,
    read_detections,
    read_manifest,
    read_tracks,
    remap_frames,
    write_detections,
    write_tracks,
)

PY = sys.executable
SIO = FIX / "synth_io"


def grid_dets(n_frames=12, shift=(1.0, 0.5), n=25, spacing=40.0, r=6.0, seed=0, noise=0.0):
    """Jittered grid translating by `shift` px/frame (d = 12 px): trivially linkable."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(np.sqrt(n)))
    g = np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2)[:n] * spacing + 50
    g = g + rng.uniform(-5, 5, g.shape)
    rows, gt = [], []
    for f in range(n_frames):
        p = g + np.array(shift) * f
        for i, (x, y) in enumerate(p):
            gt.append((f, i + 1, x, y))
            rows.append((f, x + rng.normal(0, noise), y + rng.normal(0, noise), r, 1.0))
    d = prepare_detections(pd.DataFrame(rows, columns=["frame", "x", "y", "r", "score"]))
    gtf = pd.DataFrame(gt, columns=["frame", "track_id", "x", "y"]).assign(r=0.0, missed=0)
    return d, gtf


def true_succ(d, gtf):
    gid = validate.det_gt_ids(d, gtf, 6.0)
    key = {(f, g): i for i, (f, g) in enumerate(zip(d["frame"], gid, strict=True))}
    s = np.full(len(d), -1, int)
    for (f, g), i in key.items():
        if (f + 1, g) in key:
            s[i] = key[(f + 1, g)]
    return s


# ---------------------------------------------------------------- decimation
def test_decimate_offsets_partition_frames_and_keep_det_id():
    d, _ = grid_dets(n_frames=11)
    k = 3
    seen = []
    for o in range(k):
        dd, src = decimate.decimate(d, k, o)
        assert src == list(range(o, 11, k))
        assert dd["frame"].max() == len(src) - 1
        back = np.asarray(src)[dd["frame"].to_numpy()]
        assert np.array_equal(back, d.set_index("det_id").loc[dd["det_id"], "frame"].to_numpy())
        seen += src
    assert sorted(seen) == list(range(11))
    with pytest.raises(ValueError):
        decimate.grid(0, 10, 3, 3)


def test_reverse_and_manifest_roundtrip(tmp_path):
    d, _ = grid_dets(n_frames=5)
    dr, src = decimate.reverse(d)
    assert src == [4, 3, 2, 1, 0]
    assert set(dr.loc[dr["frame"] == 0, "det_id"]) == set(d.loc[d["frame"] == 4, "det_id"])
    p = tmp_path / "rev.csv"
    decimate.write_derived(d, src, p, "reverse")
    assert read_manifest(p)["src_frames"] == src
    paths = decimate.decimate_all(d, 2, tmp_path / "dec")
    assert [read_manifest(q)["offset"] for q in paths] == [0, 1]
    assert read_detections(paths[1], keep_det_id=True)["det_id"].tolist() == \
        d.loc[d["frame"] % 2 == 1, "det_id"].tolist()


def test_decimated_video_is_lossless(tmp_path):
    import cv2
    rng = np.random.default_rng(0)
    frames = [(rng.random((24, 32, 3)) * 255).astype(np.uint8) for _ in range(6)]
    src = tmp_path / "src.avi"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (32, 24))
    for f in frames:
        vw.write(f)
    vw.release()
    out = tmp_path / "o1.avi"
    assert decimate.write_video(src, out, [1, 3, 5]) == 3
    cap = cv2.VideoCapture(str(out))
    got = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        got.append(f)
    assert len(got) == 3
    for g, i in zip(got, [1, 3, 5], strict=True):
        assert np.array_equal(g, frames[i])


# ---------------------------------------------------------------- linkers
@pytest.mark.parametrize("name", linkers.BUILTIN)
def test_builtin_linkers_exact_on_easy_motion(name):
    d, gtf = grid_dets(noise=0.2)
    n = int(d["frame"].max()) + 1
    s = linkers.link_sequence(name, d, n, len(d))
    assert np.array_equal(s, true_succ(d, gtf))


def test_linkers_do_not_import_cbtv():
    code = ("import sys; sys.path.insert(0, %r); from tools.realdata import consensus, linkers, fb, decimate; "
            "print(any(m == 'cbtv' or m.startswith('cbtv.') for m in sys.modules))" % str(ROOT))
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_snap_and_link_rows_roundtrip():
    d, gtf = grid_dets(noise=0.2)
    s = true_succ(d, gtf)
    t = linkers.link_rows(s, d, range(12))
    t2 = t.copy()
    t2["x"] += 0.7                                    # tracker-refined positions, still within the snap tolerance
    coast = t2.iloc[:1].assign(missed=2, x=1e4)       # a coasting row (predicted, not observed) is ignored
    t2 = pd.concat([t2, coast], ignore_index=True)
    t2 = t2[t2["missed"] == 0]
    s2, info = linkers.tracks_to_succ(t2, d, list(range(12)), tol=4.8)
    assert np.array_equal(s, s2)
    assert info["snapped"] == len(d)


# ---------------------------------------------------------------- forward-backward
def test_fb_consistency_handmade():
    f = np.array([2, 3, -1, -1])
    b = np.array([-1, -1, 0, 0])                     # backward maps 3 -> 0, but forward linked 1 -> 3
    assert fb.consistency(f, b).tolist() == [True, False, False, False]
    assert fb.rate(f, b) == 0.5


def test_fb_builtin_easy_is_fully_consistent():
    d, _ = grid_dets()
    s_f, s_b = fb.builtin_pair("lap", d, list(range(12)))
    assert fb.rate(s_f, s_b) == 1.0


# ---------------------------------------------------------------- consensus
def _members(d, gtf):
    s = true_succ(d, gtf)
    rev = np.full(len(s), -1, int)
    rev[s[s >= 0]] = np.nonzero(s >= 0)[0]          # a perfect backward pass maps b -> a
    return s, [consensus.Member(n, "builtin:" + n, s.copy(), rev.copy()) for n in ("a", "b", "c")]


def test_consensus_unanimity_no_majority_vote():
    d, gtf = grid_dets()
    s, ms = _members(d, gtf)
    a = int(np.nonzero(s >= 0)[0][3])
    other = int(s[np.nonzero(s >= 0)[0][4]])
    ms[2].succ[a] = other                            # 2 of 3 agree on a -> s[a]: still NOT a reference link
    ref = consensus.build(d, ms)
    assert ref.accepted[a] == -1
    lk = ref.links.set_index(["det_a", "det_b"])
    assert lk.loc[(a, s[a]), "status"] == "disagreement" and lk.loc[(a, s[a]), "reasons"] == "members 2/3"
    assert lk.loc[(a, other), "status"] == "disagreement"
    assert lk.loc[(a, other), "proposers"] == "c"
    assert ref.meta["counts"]["reference_links"] == int((s >= 0).sum()) - 1


def test_consensus_forward_backward_rejects():
    d, gtf = grid_dets()
    s, ms = _members(d, gtf)
    a = int(np.nonzero(s >= 0)[0][0])
    ms[1].succ_rev[s[a]] = -1                         # member b's backward pass loses that link
    ref = consensus.build(d, ms)
    assert ref.accepted[a] == -1
    assert ref.links.set_index("det_a").loc[a, "reasons"] == "fb"


def test_consensus_decimation_conflict_rejects_window():
    d, gtf = grid_dets()
    s, ms = _members(d, gtf)
    k = 2
    gid = validate.det_gt_ids(d, gtf, 6.0)
    key = {(f, g): i for i, (f, g) in enumerate(zip(d["frame"], gid, strict=True))}
    sk = np.full(len(d), -1, int)
    for (f, g), i in key.items():
        if (f + k, g) in key:
            sk[i] = key[(f + k, g)]
    for m in ms:
        m.succ_k = sk.copy()
    ok = consensus.build(d, ms, k=k)
    assert ok.meta["counts"]["reference_links"] == int((s >= 0).sum())
    assert ok.meta["counts"]["decim_confirmed"] == int((s >= 0).sum())
    p = key[(4, 1)]
    ms[0].succ_k[p] = key[(6, 2)]                     # one member's step-2 link contradicts the chain of particle 1
    bad = consensus.build(d, ms, k=k)
    assert bad.accepted[p] == -1 and bad.accepted[key[(5, 1)]] == -1    # both links of the window
    assert bad.accepted[key[(4, 2)]] == -1                              # particle 2's chain ending at (6, 2) too
    assert bad.accepted[key[(3, 1)]] >= 0
    # require_confirm: links no decimated run confirms are dropped
    for m in ms:
        m.succ_k = np.full(len(d), -1, int)
    rc = consensus.build(d, ms, k=k, require_confirm=True)
    assert rc.meta["counts"]["reference_links"] == 0


def test_reference_write_load_and_tamper(tmp_path):
    d, gtf = grid_dets()
    _, ms = _members(d, gtf)
    out = consensus.write(consensus.build(d, ms), d, tmp_path / "ref")
    t, links, meta = consensus.load(out)
    assert meta["schema"] == consensus.SCHEMA and not meta["cbtv_derived"]
    assert t["track_id"].nunique() == 25 and len(t) == len(d)
    with open(out / "reference_tracks.csv", "a") as f:
        f.write("11,999,1.0,1.0,6.0,0\n")
    with pytest.raises(consensus.CircularityError, match="sha256"):
        consensus.load(out)
    (out / "reference.meta.json").unlink()
    with pytest.raises(consensus.CircularityError, match="provenance unknown"):
        consensus.load(out)


def test_reference_flag_inconsistency_refused(tmp_path):
    d, gtf = grid_dets()
    _, ms = _members(d, gtf)
    ms[0].cbtv_derived = True
    out = consensus.write(consensus.build(d, ms), d, tmp_path / "ref")
    mp = out / "reference.meta.json"
    m = json.loads(mp.read_text())
    m["cbtv_derived"] = False                          # hand-edited to hide a CBTV member
    mp.write_text(json.dumps(m))
    with pytest.raises(consensus.CircularityError, match="inconsistent"):
        consensus.load(out)


# ---------------------------------------------------------------- circularity guard (on synth_io, CBTV output)
@pytest.fixture(scope="module")
def sio():
    return read_detections(SIO / "detections.csv")


def test_guard_refuses_cbtv_against_cbtv_reference(sio, tmp_path):
    cbtv_out = SIO / "cpu_baseline_tracks.csv"      # CBTV ship-v4 CPU output on synth_io
    ms = consensus.builtin_members(sio, ("nn",), fb=False)
    ms.append(consensus.external_member("cbtv", "cbtv ship-v4", sio, (cbtv_out, None)))
    ref = consensus.write(consensus.build(sio, ms), sio, tmp_path / "ref")
    _, _, meta = consensus.load(ref)
    assert meta["cbtv_derived"] and [m["cbtv_derived"] for m in meta["members"]] == [False, True]
    with pytest.raises(consensus.CircularityError, match="built from CBTV output"):
        consensus.check_scoring(meta, "CBTV", 4)
    # a renamed tracker whose output carries CBTV fingerprints (coasting rows) is still refused
    other = tmp_path / "renamed.csv"
    other.write_bytes((SIO / "golden_gpu_tracks_bt.csv").read_bytes())
    with pytest.raises(consensus.CircularityError, match="fingerprint"):
        consensus.check_scoring(meta, "mytracker", 4, [other])
    # a non-CBTV tracker may be scored against it
    clean = tmp_path / "relax.csv"
    write_tracks(clean, linkers.link_rows(linkers.link_sequence("relax", sio, 60, len(sio)), sio, range(60)))
    consensus.check_scoring(meta, "relax", 4, [clean])


def test_guard_external_member_fingerprint_marks_cbtv(sio, tmp_path):
    p = tmp_path / "tp.csv"
    p.write_bytes((SIO / "cpu_baseline_tracks.csv").read_bytes())
    m = consensus.external_member("trackpy", "trackpy-nvp", sio, (p, None))
    assert m.cbtv_derived                              # declared trackpy, but the file has CBTV coasting rows


def test_guard_allows_cbtv_against_cbtv_free_reference_and_blocks_self_inputs(sio, tmp_path):
    ms = consensus.builtin_members(sio, ("nn", "lap"), k=2, fb=False)
    ref = consensus.write(consensus.build(sio, ms, k=2), sio, tmp_path / "ref")
    _, _, meta = consensus.load(ref)
    assert not meta["cbtv_derived"]
    consensus.check_scoring(meta, "cbtv", 1, [SIO / "cpu_baseline_tracks.csv"])     # allowed
    with pytest.raises(consensus.CircularityError, match="member at that step"):
        consensus.check_scoring(meta, "nn", 2)
    consensus.check_scoring(meta, "nn", 4)
    # scoring a file that is itself an input of the reference
    ext = tmp_path / "ext.csv"
    write_tracks(ext, linkers.link_rows(linkers.link_sequence("relax", sio, 60, len(sio)), sio, range(60)))
    ms2 = ms + [consensus.external_member("relaxext", "relaxation", sio, (ext, None))]
    ref2 = consensus.write(consensus.build(sio, ms2), sio, tmp_path / "ref2")
    _, _, meta2 = consensus.load(ref2)
    with pytest.raises(consensus.CircularityError, match="own inputs"):
        consensus.check_scoring(meta2, "something", 3, [ext])


def test_cli_score_refuses_with_exit_3(sio, tmp_path):
    ref = tmp_path / "ref"
    r = subprocess.run([PY, "-m", "tools.realdata", "consensus", "--detections", str(SIO / "detections.csv"),
                        "--out", str(ref), "--builtin", "nn", "--no-fb",
                        "--member", "cbtv=%s" % (SIO / "cpu_baseline_tracks.csv")],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["cbtv_derived"] is True
    r = subprocess.run([PY, "-m", "tools.realdata", "score", "--reference", str(ref), "--tracker", "cbtv",
                        "--tracks", str(SIO / "golden_gpu_tracks.csv")], cwd=str(ROOT), capture_output=True,
                       text=True)
    assert r.returncode == 3 and "REFUSED" in r.stderr
    clean = tmp_path / "ref_clean"
    subprocess.run([PY, "-m", "tools.realdata", "consensus", "--detections", str(SIO / "detections.csv"),
                    "--out", str(clean), "--builtin", "nn,lap", "--no-fb"], cwd=str(ROOT), check=True,
                   capture_output=True)
    r = subprocess.run([PY, "-m", "tools.realdata", "score", "--reference", str(clean), "--tracker", "cbtv",
                        "--tracks", str(SIO / "cpu_baseline_tracks.csv")], cwd=str(ROOT), capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res["step"] == 1 and res["klinks"] > 5000 and 0 <= res["id_err"] < 0.05


# ---------------------------------------------------------------- scoring
def test_score_perfect_and_one_switch():
    _, gtf = grid_dets()
    frames = list(range(12))
    s = score.score(gtf, gtf, frames, 6.0)
    assert s["id_err"] == 0 and s["pos_err"] == 0 and s["klinks"] == 25 * 11 and s["switched"] == 0
    t = gtf.copy()
    m1 = (t["track_id"] == 1) & (t["frame"] >= 6)
    m2 = (t["track_id"] == 2) & (t["frame"] >= 6)
    t.loc[m1, "track_id"], t.loc[m2, "track_id"] = 2, 1          # swap ids of two tracks from frame 6
    s = score.score(t, gtf, frames, 6.0)
    assert s["switched"] == 2 and s["idsw"] == 2 and s["correct"] == 25 * 11 - 2
    # at step 2 (offsets pooled) the swap is seen once per offset grid that straddles frame 6
    runs = [(t[t["frame"] % 2 == o], list(range(o, 12, 2))) for o in range(2)]
    s2 = score.score_offsets(runs, gtf, 6.0)
    assert s2["klinks"] == 25 * (5 + 5) and s2["switched"] == 4


def test_score_unmatched_counts_as_missed_not_switch():
    _, gtf = grid_dets()
    t = gtf[~((gtf["track_id"] == 3) & (gtf["frame"] == 5))]
    s = score.score(t, gtf, list(range(12)), 6.0)
    assert s["missed"] == 2 and s["switched"] == 0


# ---------------------------------------------------------------- audit
def test_cohen_kappa_and_wilson():
    a = ["same"] * 20 + ["different"] * 5 + ["same"] * 3 + ["different"] * 2
    b = ["same"] * 20 + ["different"] * 5 + ["different"] * 3 + ["same"] * 2
    po = 25 / 30
    pe = (23 / 30) * (22 / 30) + (7 / 30) * (8 / 30)
    assert audit.cohen_kappa(a, b) == pytest.approx((po - pe) / (1 - pe))
    assert audit.cohen_kappa(["same"] * 4, ["same"] * 4) == 1.0
    lo, hi = audit.wilson(0, 300)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0126, abs=2e-4)
    assert audit.normalise(" Same ") == "same" and audit.normalise(np.nan) == ""


def _synthetic_reference(tmp_path):
    g, det, col = synth.make(n=40, frames=40, size=(200, 200), r=5.0, disp=0.06, seed=3)
    d = prepare_detections(det)
    gt = g.rename(columns={"gt_id": "track_id"})[["frame", "track_id", "x", "y"]].assign(r=0.0, missed=0)
    ms = consensus.builtin_members(d, ("nn", "lap"), k=2)
    ref = consensus.write(consensus.build(d, ms, k=2), d, tmp_path / "ref")
    return d, gt, g, col, ref


def test_audit_sample_is_stratified_and_blind(tmp_path):
    d, gt, g, col, ref = _synthetic_reference(tmp_path)
    _, links, _ = consensus.load(ref)
    key, st = audit.sample(links, d, 60, seed=0)
    n_dis = int((links["status"] == "disagreement").sum())
    assert (key["kind"] == "disagreement").sum() == n_dis
    assert (st["sampled"] >= 1).all() and (st["sampled"] <= st["population"]).all()
    assert key["item_id"].is_unique and key["item_id"].str.fullmatch("[0-9a-f]{8}").all()
    clip = tmp_path / "clip.avi"
    synth.render(g, col, clip, size=(200, 200), r=5.0)
    out = audit.export(key, st, tmp_path / "audit", video=clip, d_px=10.0)
    a = pd.read_csv(out / "annotator_A.csv", dtype={"item_id": str})
    b = pd.read_csv(out / "annotator_B.csv", dtype={"item_id": str})
    assert list(a.columns) == ["item_id", "image", "judgement", "comment"]        # nothing that hints the source
    assert set(a["item_id"]) == set(key["item_id"]) == set(b["item_id"])
    assert a["item_id"].tolist() != b["item_id"].tolist()
    assert len(list((out / "crops").glob("*.png"))) == len(key)


def test_crop_pair_draws_only_in_margin():
    rng = np.random.default_rng(0)
    fa = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    fb_ = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    img = audit.crop_pair(fa, fb_, 40, 30, 43, 31, half=10, scale=2, margin=6)
    s = 21 * 2
    left = img[6:6 + s, 6:6 + s]
    assert np.array_equal(left[::2, ::2], fa[20:41, 30:51])
    right = img[6:6 + s, (s + 12) + 8 + 6:(s + 12) + 8 + 6 + s]
    assert np.array_equal(right[::2, ::2], fb_[20:41, 30:51])


def test_audit_report_and_apply(tmp_path):
    d, gt, _, _, ref = _synthetic_reference(tmp_path)
    _, links, meta = consensus.load(ref)
    key, st = audit.sample(links, d, 80, seed=0)
    out = audit.export(key, st, tmp_path / "audit")
    gid = validate.det_gt_ids(d, gt, 5.0)
    truth = np.where((gid[key["det_a"]] >= 0) & (gid[key["det_a"]] == gid[key["det_b"]]), "same", "different")
    lab = dict(zip(key["item_id"], truth, strict=True))
    a = pd.read_csv(out / "annotator_A.csv", dtype={"item_id": str})
    b = pd.read_csv(out / "annotator_B.csv", dtype={"item_id": str})
    a["judgement"] = a["item_id"].map(lab)
    b["judgement"] = b["item_id"].map(lab)
    flip = b.index[:5]                                # annotator B disagrees on 5 items
    b.loc[flip, "judgement"] = b.loc[flip, "judgement"].map({"same": "different", "different": "same"})
    a.to_csv(out / "annotator_A.csv", index=False)
    b.to_csv(out / "annotator_B.csv", index=False)
    rep = audit.report(key, st, a, b, smallest_effect=0.05)
    assert rep["judged_by_both"] == len(key)
    assert rep["raw_agreement"] == pytest.approx(1 - 5 / len(key))
    assert 0.5 < rep["kappa"] < 1.0
    q = validate.reference_quality(links, d, gt, 5.0)
    assert rep["reference_errors"] <= q["ref_wrong"]
    assert rep["ref_error_ci95"][0] <= rep["ref_error_rate"] <= rep["ref_error_ci95"][1]
    assert "accepted" in rep
    new = audit.apply(ref, d, out, tmp_path / "ref_audited")
    _, links2, meta2 = consensus.load(new)
    both_same_dis = sum(1 for i, k in zip(key["item_id"], key["kind"], strict=True)
                        if k == "disagreement" and lab[i] == "same" and i not in set(b.loc[flip, "item_id"]))
    assert meta2["counts"]["reference_links"] >= meta["counts"]["reference_links"]
    assert meta2["audit"]["added"] + meta2["audit"]["conflicts"] == both_same_dis
    assert meta2["cbtv_derived"] == meta["cbtv_derived"]
    q2 = validate.reference_quality(links2, d, gt, 5.0)
    assert q2["coverage"] >= q["coverage"]


# ---------------------------------------------------------------- synthetic generator + validator
def test_synth_make_deterministic_and_slow():
    g1, d1, _ = synth.make(n=50, frames=30, seed=7)
    g2, d2, _ = synth.make(n=50, frames=30, seed=7)
    assert g1.equals(g2) and d1.equals(d2)
    disp = validate.gt_displacement(g1.rename(columns={"gt_id": "track_id"})) / 12.0
    assert 0.01 < disp < 0.08                         # "tiny displacement" window: <= ~0.05 d/frame on average
    assert 0.9 < len(d1) / len(g1) < 1.05


def test_validator_pseudo_gt_tracks_true_error(tmp_path):
    """End-to-end on a short synthetic clip: CBTV-free reference has no wrong links, and the audited pseudo-GT
    measures a decimated tracker's ID-switch rate within 1 point of the true-GT value."""
    d, gt, _, _ = validate.highfps_data(frames=96)
    q, rows = validate.evaluate("hf", d, gt, 4, [8, 16], tmp_path, cells=("relax",), n_audit=100)
    assert q["ref_err"] < 0.002 and q["audited_coverage"] > 0.99
    for r in rows:
        assert abs(r["audited_id_err"] - r["true_id_err"]) < 0.01, r
        assert r["pseudo_id_err"] <= r["true_id_err"] + 0.01
    assert rows[1]["true_id_err"] > rows[0]["true_id_err"]          # harder at the larger step, and it shows


def test_validator_on_synth_io_reference_is_exact(tmp_path):
    d = read_detections(SIO / "detections.csv")
    gt = validate.read_gt(SIO / "gt.csv")
    ms = consensus.builtin_members(d, ("relax", "nn", "lap"), k=2)
    ref = consensus.write(consensus.build(d, ms, k=2), d, tmp_path / "ref")
    _, links, _ = consensus.load(ref)
    q = validate.reference_quality(links, d, gt, 8.0)
    assert q["ref_wrong"] == 0 and q["coverage"] > 0.99
    t = remap_frames(read_tracks(SIO / "cpu_baseline_tracks.csv"), list(range(60)))
    rt, _, _ = consensus.load(ref)
    sp = score.score(t, rt, list(range(60)), 8.0)
    st = score.score(t, gt, list(range(60)), 8.0)
    assert abs(sp["id_err"] - st["id_err"]) < 0.005


def test_cli_decimate_link_fb(tmp_path):
    d, _ = grid_dets()
    p = tmp_path / "dets.csv"
    write_detections(p, d, keep_det_id=False)
    r = subprocess.run([PY, "-m", "tools.realdata", "decimate", "--detections", str(p), "--k", "3", "--out",
                        str(tmp_path / "dec")], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "dec").glob("dets_k3_o*.csv"))) == 3
    r = subprocess.run([PY, "-m", "tools.realdata", "fb", "--detections", str(p), "--linker", "nn"], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert json.loads(r.stdout)["fb_rate"] == 1.0
    r = subprocess.run([PY, "-m", "tools.realdata", "link", "--detections", str(p), "--linker", "relax", "--out",
                        str(tmp_path / "t.csv")], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert read_tracks(tmp_path / "t.csv")["track_id"].nunique() == 25
