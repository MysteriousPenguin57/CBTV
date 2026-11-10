"""Validate the validator: build the CBTV-free pseudo-GT where the true GT is known and compare, per tracker cell,
the error measured against the pseudo-GT with the error measured against the true GT.

Datasets: tests/fixtures/synth_io (30 fps, ~0.14 d/frame, detection recall ~0.86) and a generated high-fps clip
(``synth.make``, ~0.03 d/frame). Tracker cells: the builtin linkers on decimated sequences (steps the reference did
not use), and optionally CBTV (``--cbtv``: runs the tracker on lossless decimated clips; slow).

  python -m tools.realdata validate --out runs/c015_validate [--cbtv]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit, consensus, score, synth
from .decimate import decimate, decimate_all, frame_range
from .linkers import diameter, link_rows, link_sequence
from .score import match_frame
from .tracksio import prepare_detections, read_detections, read_gt, read_tracks, remap_frames

ROOT = Path(__file__).resolve().parents[2]
SYNTH_IO = ROOT / "tests" / "fixtures" / "synth_io"


def det_gt_ids(d: pd.DataFrame, gt: pd.DataFrame, gate: float) -> np.ndarray:
    """GT id per det_id (-1 = false positive / unmatched), one-to-one per frame within gate."""
    out = np.full(len(d), -1, int)
    gb = {int(f): g for f, g in gt.groupby("frame", sort=False)}
    for f, g in d.groupby("frame", sort=False):
        t = gb.get(int(f))
        if t is None:
            continue
        i, j, _ = match_frame(g[["x", "y"]].to_numpy(float), t[["x", "y"]].to_numpy(float), gate)
        out[g["det_id"].to_numpy()[i]] = t["track_id"].to_numpy()[j]
    return out


def reference_quality(links: pd.DataFrame, d: pd.DataFrame, gt: pd.DataFrame, gate: float) -> dict:
    """Error rate of the reference links vs truth, coverage of truly linkable detection pairs, and how many of the
    discarded disagreement links were in fact correct."""
    gid = det_gt_ids(d, gt, gate)
    ref = links[links["status"] == "reference"]
    dis = links[links["status"] == "disagreement"]
    ga, gb = gid[ref["det_a"].to_numpy()], gid[ref["det_b"].to_numpy()]
    wrong = int(((ga < 0) | (ga != gb)).sum())
    # truly linkable: consecutive-frame det pairs carrying the same GT id
    fr = d["frame"].to_numpy()
    key = {(int(f), int(g)): int(i) for i, (f, g) in enumerate(zip(fr, gid, strict=True)) if g >= 0}
    linkable = sum(1 for (f, g) in key if (f + 1, g) in key)
    da, db = gid[dis["det_a"].to_numpy()], gid[dis["det_b"].to_numpy()]
    dis_true = int(((da >= 0) & (da == db)).sum())
    xy = d[["x", "y"]].to_numpy(float)
    on = gid >= 0
    gpos = gt.set_index(["frame", "track_id"])[["x", "y"]]
    idx = pd.MultiIndex.from_arrays([fr[on], gid[on]])
    perr = np.linalg.norm(xy[on] - gpos.loc[idx].to_numpy(float), axis=1)
    return {"ref_links": int(len(ref)), "ref_wrong": wrong, "ref_err": wrong / len(ref) if len(ref) else float("nan"),
            "linkable": int(linkable), "coverage": (len(ref) - wrong) / linkable if linkable else float("nan"),
            "disagreements": int(len(dis)), "disagreements_true": dis_true, "det_pos_err": float(perr.mean())}


def builtin_cell(name: str, d: pd.DataFrame, k: int):
    """[(tracks on source frames, evaluated source frames)] for offsets 0..k-1 of a builtin linker."""
    runs = []
    for o in range(k):
        dd, src = decimate(d, k, o)
        succ = link_sequence(name, dd, len(src), len(d))
        runs.append((link_rows(succ, d, src), src))
    return runs


def run_cbtv(clip, dets_csv, out_dir, r_min=6, r_max=10, preset="ship-v4") -> Path:
    """CPU CBTV run (same determinism env and flags as tests/conftest.run_fixture) on a clip + detections csv."""
    import run_cbtv as rc
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", OPENCV_OPENCL_RUNTIME="disabled",
               PYTHONHASHSEED="0")
    env.pop("CBTV_DUMP_C", None)
    argv = [sys.executable, "-u", "-m", "cbtv.run_tracker", "--input", str(clip),
            "--output-video", str(out_dir / "tracks.mp4"), "--output-csv", str(out_dir / "tracks.csv"),
            "--frame-step", "1", "--detector", "csv", "--detections-csv", str(dets_csv),
            "--use-gpu", "0", "--gpu-device", "0", "--r-min", str(r_min), "--r-max", str(r_max)]
    argv += list(rc.PRESETS[preset])
    r = subprocess.run(argv, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError("cbtv failed rc=%d\n%s" % (r.returncode, r.stdout.decode("utf-8", "replace")[-2000:]))
    try:
        (out_dir / "tracks.mp4").unlink()
    except OSError:
        pass
    return out_dir / "tracks.csv"


def cbtv_cell(d: pd.DataFrame, k: int, video, work, full_tracks=None, r_range=(6, 10)):
    """CBTV at step k: decimated lossless clips + detections per offset, run, map back to source frames."""
    if k == 1 and full_tracks is not None:
        f0, f1 = frame_range(d)
        src = list(range(f0, f1 + 1))
        return [(remap_frames(read_tracks(full_tracks), src), src)], [Path(full_tracks)]
    work = Path(work) / ("cbtv_k%d" % k)
    paths = decimate_all(d, k, work, video=video)
    runs, files = [], []
    for o, p in enumerate(paths):
        t = run_cbtv(p.with_suffix(".avi"), p, work / ("o%d" % o), *r_range)
        src = json.loads(p.with_name(p.stem + ".seq.json").read_text())["src_frames"]
        runs.append((remap_frames(read_tracks(t), src), src))
        files.append(t)
    return runs, files


def evaluate(dataset: str, d: pd.DataFrame, gt: pd.DataFrame, k_ref: int, steps, out_dir, video=None,
             cbtv_steps=(), cbtv_full=None, builtin=("relax", "nn", "lap"), cells=("relax", "nn", "lap"),
             n_audit: int = 300, cbtv_r=(6, 10)):
    """Build the CBTV-free reference, then score every cell against it and against the true GT."""
    out_dir = Path(out_dir) / dataset
    dpx = diameter(d)
    gate = 0.5 * dpx
    src_info = {"path": dataset, "sha256": ""}
    members = consensus.builtin_members(d, builtin, k=k_ref, fb=True, det_source=src_info)
    ref = consensus.build(d, members, k=k_ref, det_source=src_info)
    rdir = consensus.write(ref, d, out_dir / "reference")
    ref_tracks, links, meta = consensus.load(rdir)
    q = reference_quality(links, d, gt, gate)
    q.update(dataset=dataset, k_ref=k_ref, d_px=dpx, gate=gate, counts=meta["counts"])
    # blinded audit with simulated perfect annotators (GT oracle): measures the reference error rate the way the
    # real audit will, then folds the verdicts on the disagreements back into the reference
    key, st = audit.sample(links, d, n_audit, seed=1)
    adir = audit.export(key, st, out_dir / "audit", video=None, d_px=dpx)
    gid = det_gt_ids(d, gt, gate)
    oracle = np.where((gid[key["det_a"].to_numpy()] >= 0)
                      & (gid[key["det_a"].to_numpy()] == gid[key["det_b"].to_numpy()]), "same", "different")
    for who in ("A", "B"):
        ann = pd.read_csv(adir / ("annotator_%s.csv" % who), dtype={"item_id": str})
        ann["judgement"] = ann["item_id"].map(dict(zip(key["item_id"], oracle, strict=True)))
        ann.to_csv(adir / ("annotator_%s.csv" % who), index=False, lineterminator="\n")
    rep = audit.report(key, st, *(pd.read_csv(adir / ("annotator_%s.csv" % w), dtype={"item_id": str})
                                  for w in ("A", "B")))
    q.update(audit_items=rep["reference_items"], audit_err=rep["ref_error_rate"], audit_ci=rep["ref_error_ci95"],
             audit_kappa=rep["kappa"], audit_dis_same=rep["disagreements_same"])
    a2 = audit.apply(rdir, d, adir, out_dir / "reference_audited")
    ref2_tracks, links2, _ = consensus.load(a2)
    q2 = reference_quality(links2, d, gt, gate)
    q.update(audited_ref_links=q2["ref_links"], audited_ref_err=q2["ref_err"], audited_coverage=q2["coverage"])
    rows = []
    jobs = [(c, k) for c in cells for k in steps] + [("cbtv", k) for k in cbtv_steps]
    disp = gt_displacement(gt) / dpx
    for name, k in jobs:
        if name == "cbtv":
            runs, files = cbtv_cell(d, k, video, out_dir, cbtv_full, cbtv_r)
        else:
            runs, files = builtin_cell(name, d, k), []
        consensus.check_scoring(meta, name, k, files)            # the guard: raises if circular
        row = {"dataset": dataset, "tracker": name, "step": k, "disp_d": disp * k}
        for tag, r in (("true", gt), ("pseudo", ref_tracks), ("audited", ref2_tracks)):
            sc = score.score_offsets(runs, r, gate)
            row.update({tag + "_" + f: sc[f] for f in score.FIELDS})
        rows.append(row)
    return q, rows


def gt_displacement(gt: pd.DataFrame) -> float:
    g = gt.sort_values(["track_id", "frame"])
    same = (g["track_id"].to_numpy()[1:] == g["track_id"].to_numpy()[:-1]) & \
           (np.diff(g["frame"].to_numpy()) == 1)
    dxy = np.diff(g[["x", "y"]].to_numpy(float), axis=0)[same]
    return float(np.linalg.norm(dxy, axis=1).mean())


def synth_io_data():
    d = read_detections(SYNTH_IO / "detections.csv")
    gt = read_gt(SYNTH_IO / "gt.csv")
    return d, gt


def highfps_data(seed: int = 0, frames: int = 240):
    g, det, colours = synth.make(frames=frames, seed=seed)
    d = prepare_detections(det)
    gt = g[g["visible"] == 1].rename(columns={"gt_id": "track_id"})[["frame", "track_id", "x", "y"]].copy()
    gt["r"] = 0.0
    gt["missed"] = 0
    return d, gt.reset_index(drop=True), g, colours


def fmt(x, nd=3):
    if isinstance(x, float):
        return "nan" if np.isnan(x) else ("%.*f" % (nd, x))
    return str(x)


def table(qs, rows) -> str:
    lines = ["Reference: CBTV-free consensus (relax + nn + lap, unanimous, forward-backward, decimation consensus "
             "at k_ref), vs true GT. Audit = blinded sample (GT oracle as both annotators) + all disagreements.", "",
             "| dataset | k_ref | ref links | ref link error (wrong) | coverage | disagreements (truly correct) "
             "| audit est. error [95% CI] (n) | audited: ref links, error, coverage | det pos err px |",
             "|---|---|---|---|---|---|---|---|---|"]
    for q in qs:
        lines.append("| %s | %d | %d | %s (%d) | %s | %d (%d) | %s [%s, %s] (%d) | %d, %s, %s | %s |" % (
            q["dataset"], q["k_ref"], q["ref_links"], fmt(q["ref_err"], 4), q["ref_wrong"], fmt(q["coverage"], 4),
            q["disagreements"], q["disagreements_true"], fmt(q["audit_err"], 4), fmt(q["audit_ci"][0], 4),
            fmt(q["audit_ci"][1], 4), q["audit_items"], q["audited_ref_links"], fmt(q["audited_ref_err"], 4),
            fmt(q["audited_coverage"], 4), fmt(q["det_pos_err"])))
    lines += ["", "Tracker cells: ID-switch rate per k-link (switched / (correct + switched)) and switched count, "
              "measured vs true GT | vs pseudo-GT | vs audited pseudo-GT; pos = mean matched distance px.", "",
              "| dataset | tracker | step | disp d/step | id_err true | id_err pseudo | id_err audited "
              "| switches true / pseudo / audited | k-links true / pseudo | pos err true | pos err pseudo |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %d / %d / %d | %d / %d | %s | %s |" % (
            r["dataset"], r["tracker"], r["step"], fmt(r["disp_d"], 2), fmt(r["true_id_err"], 4),
            fmt(r["pseudo_id_err"], 4), fmt(r["audited_id_err"], 4), r["true_switched"], r["pseudo_switched"],
            r["audited_switched"], r["true_klinks"], r["pseudo_klinks"], fmt(r["true_pos_err"]),
            fmt(r["pseudo_pos_err"])))
    return "\n".join(lines)


def main(out, cbtv: bool = False, fast: bool = False) -> str:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qs, rows = [], []
    d, gt = synth_io_data()
    q, r = evaluate("synth_io", d, gt, 2, [3, 4, 6, 8] if not fast else [4], out,
                    video=SYNTH_IO / "clip.avi", cbtv_steps=(1, 2, 4) if cbtv else (),
                    cbtv_full=SYNTH_IO / "cpu_baseline_tracks.csv")
    qs.append(q); rows += r
    d, gt, g, colours = highfps_data(frames=120 if fast else 240)
    video = None
    if cbtv:
        video = out / "highfps.avi"
        synth.render(g, colours, video)
    q, r = evaluate("highfps", d, gt, 4, [8, 16, 32] if not fast else [8, 32], out, video=video,
                    cbtv_steps=(1, 8, 16) if cbtv else (), cbtv_r=(4, 8))
    qs.append(q); rows += r
    txt = table(qs, rows)
    (out / "validation_table.md").write_text(txt + "\n")
    pd.DataFrame(rows).to_csv(out / "validation_cells.csv", index=False, lineterminator="\n")
    (out / "reference_quality.json").write_text(json.dumps(qs, indent=1, default=str) + "\n")
    return txt
