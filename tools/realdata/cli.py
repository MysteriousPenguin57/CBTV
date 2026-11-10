"""Command line for the realdata tools (route A). ``python -m tools.realdata <command> -h`` for details.

  decimate      dets.csv -> dets_k{k}_o{o}.csv (+ .seq.json, + lossless .avi with --video) for offsets 0..k-1
  reverse       dets.csv -> reversed detections (+ clip) for the backward pass of an external tracker
  link          run a CBTV-free builtin linker (relax | nn | lap) on a detections csv
  fb            forward-backward agreement rate (builtin linker, or forward + backward tracks of any tracker)
  consensus     build the pseudo-GT reference (+ provenance) from builtin and/or external members
  score         score tracker output against a reference (refuses circular scoring, exit code 3)
  audit-sample  blinded audit package (crops + two annotator csvs + sealed key)
  audit-report  Cohen's kappa + reference error rate (95% CI) from the two filled csvs (Amendment 2: HT estimates)
  audit-adjudicate  csv of the A-B disagreements on the overlap, to adjudicate jointly
  audit-apply   fold the audit verdicts into a new reference
  validate      the validation table on synth_io + a generated high-fps clip (pseudo-GT vs true GT)

A tracks argument is TRACKS[@SEQ_CSV]: SEQ_CSV is the derived detections csv the tracker ran on (its .seq.json
manifest maps processed frames to source frames); without it the run is taken as full rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import audit, consensus, decimate, fb, linkers, score
from .tracksio import read_detections, read_manifest, read_tracks, remap_frames, sha256_file, write_tracks

EXIT_REFUSED = 3


def _split(spec: str):
    p, _, s = spec.partition("@")
    return p, (s or None)


def _named(spec: str):
    """NAME[:PRODUCER]=TRACKS[@SEQ] -> (name, producer, (tracks, seq))."""
    if "=" not in spec:
        raise SystemExit("member spec must be NAME[:PRODUCER]=TRACKS[@SEQ_CSV], got %r" % spec)
    lhs, rhs = spec.split("=", 1)
    name, _, producer = lhs.partition(":")
    return name, producer or name, _split(rhs)


def cmd_decimate(a):
    d = read_detections(a.detections)
    paths = decimate.decimate_all(d, a.k, a.out, video=a.video, source=str(a.detections))
    print("\n".join(str(p) for p in paths))


def cmd_reverse(a):
    d = read_detections(a.detections)
    _, src = decimate.reverse(d)
    decimate.write_derived(d, src, a.out, "reverse", video=a.video, source=str(a.detections))
    print(a.out)


def cmd_link(a):
    d = read_detections(a.detections)
    n = int(d["frame"].max()) + 1 if len(d) else 0
    succ = linkers.link_sequence(a.linker, d, n, len(d))
    write_tracks(a.out, linkers.link_rows(succ, d, range(n)))
    print("%s: %d links -> %s" % (a.linker, int((succ >= 0).sum()), a.out))


def cmd_fb(a):
    d = read_detections(a.detections)
    f0, f1 = decimate.frame_range(d)
    if a.linker:
        s_f, s_b = fb.builtin_pair(a.linker, d, list(range(f0, f1 + 1)))
    else:
        if not (a.forward and a.backward):
            raise SystemExit("fb needs --linker, or --forward and --backward TRACKS[@SEQ]")
        tol = 0.4 * linkers.diameter(d)
        sides = []
        for spec, default in ((a.forward, list(range(f0, f1 + 1))), (a.backward, list(range(f1, f0 - 1, -1)))):
            t, seq = _split(spec)
            src = read_manifest(seq)["src_frames"] if seq else default
            s, _ = linkers.tracks_to_succ(remap_frames(read_tracks(t), src), d, src, tol)
            sides.append(s)
        s_f, s_b = sides
    print(json.dumps({"forward_links": int((s_f >= 0).sum()), "fb_rate": fb.rate(s_f, s_b)}))


def cmd_consensus(a):
    d = read_detections(a.detections)
    det_src = {"path": str(a.detections), "sha256": sha256_file(a.detections)}
    names = [x for x in (a.builtin or "").split(",") if x]
    members = consensus.builtin_members(d, names, k=a.k, fb=not a.no_fb, det_source=det_src) if names else []
    rev = {n: s for n, _, s in map(_named, a.member_rev or [])}
    dec: dict = {}
    for n, _, s in map(_named, a.member_dec or []):
        dec.setdefault(n, []).append(s)
    for n, prod, fwd in map(_named, a.member or []):
        members.append(consensus.external_member(n, prod, d, fwd, rev.get(n), dec.get(n, ()), k=a.k))
    if not members:
        raise SystemExit("no members: give --builtin and/or --member")
    ref = consensus.build(d, members, k=a.k, require_confirm=a.require_confirm, det_source=det_src)
    out = consensus.write(ref, d, a.out)
    m = json.loads((out / "reference.meta.json").read_text())
    print(json.dumps({"out": str(out), "cbtv_derived": m["cbtv_derived"], **m["counts"]}, indent=1))


def cmd_score(a):
    try:
        ref_tracks, _, meta = consensus.load(a.reference)
        runs, files, steps = [], [], set()
        for spec in a.tracks:
            t, seq = _split(spec)
            files.append(t)
            if seq:
                info = read_manifest(seq)
                src, k = info["src_frames"], int(info.get("k", 1))
                if info.get("kind") == "reverse":
                    raise SystemExit("score a forward run, not a reversed one")
            else:
                f0, f1 = int(ref_tracks["frame"].min()), int(ref_tracks["frame"].max())
                src, k = list(range(f0, f1 + 1)), 1
            steps.add(k)
            runs.append((remap_frames(read_tracks(t), src), src))
        if len(steps) != 1:
            raise SystemExit("all --tracks of one score call must share one step (got %s)" % sorted(steps))
        step = steps.pop()
        consensus.check_scoring(meta, a.tracker, step, files)
    except consensus.CircularityError as e:
        print("REFUSED: %s" % e, file=sys.stderr)
        return EXIT_REFUSED
    gate = a.gate or float(pd.read_csv(Path(a.reference) / "reference_tracks.csv")["r"].median())  # 0.5 d
    res = score.score_offsets(runs, ref_tracks, gate)
    print(json.dumps({"tracker": a.tracker, "step": step, "gate_px": gate, "reference_cbtv_derived":
                      meta["cbtv_derived"], **res}, indent=1))
    return 0


def cmd_audit_sample(a):
    _, links, _ = consensus.load(a.reference)
    d = read_detections(a.detections)
    if a.design == "legacy":
        seed = 0 if a.seed is None else a.seed
        key, st = audit.sample(links, d, a.n, seed=seed, lane_bins=a.lane_bins, phase_bins=a.phase_bins,
                               density_bins=a.density_bins)
    else:                                               # Amendment 2
        seed = audit.A2_SEED if a.seed is None else a.seed
        key, st = audit.sample_a2(links, d, n_random=a.n_random, n_dis=a.n_dis, seed=seed, b_frac=a.b_frac)
    out = audit.export(key, st, a.out, video=a.video, d_px=linkers.diameter(d), seed=seed,
                       frame_offset=a.frame_offset)
    if a.design != "legacy":
        (Path(out) / "DESIGN.txt").write_text(audit.design_note(st, a.n_random, a.n_dis))
    print("%d items (%d reference, %d disagreement) -> %s" % (
        len(key), int((key["kind"] == "reference").sum()), int((key["kind"] == "disagreement").sum()), out))


def cmd_audit_report(a):
    ad = Path(a.audit)
    rd = lambda p: pd.read_csv(p, dtype={"item_id": str})  # noqa: E731
    key, st = rd(ad / "audit_key.csv"), pd.read_csv(ad / "audit_strata.csv")
    ann_a, ann_b = rd(a.a or ad / "annotator_A.csv"), rd(a.b or ad / "annotator_B.csv")
    if "pi" in key.columns:                              # Amendment 2 key
        adj = rd(a.adjudication) if a.adjudication else None
        rep = audit.report_a2(key, st, ann_a, ann_b, adj, n_boot=a.n_boot)
    else:
        rep = audit.report(key, st, ann_a, ann_b, a.smallest_effect)
    print(json.dumps(rep, indent=1))


def cmd_audit_adjudicate(a):
    ad = Path(a.audit)
    rd = lambda p: pd.read_csv(p, dtype={"item_id": str})  # noqa: E731
    out = audit.adjudication_sheet(rd(ad / "audit_key.csv"), rd(a.a or ad / "annotator_A.csv"),
                                   rd(a.b or ad / "annotator_B.csv"), a.out or ad / "adjudication.csv")
    print(out)


def cmd_audit_apply(a):
    d = read_detections(a.detections)
    print(audit.apply(a.reference, d, a.audit, a.out, a.a, a.b))


def cmd_validate(a):
    from . import validate
    print(validate.main(a.out, cbtv=a.cbtv, fast=a.fast))


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m tools.realdata", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decimate"); p.set_defaults(fn=cmd_decimate)
    p.add_argument("--detections", required=True); p.add_argument("--k", type=int, required=True)
    p.add_argument("--out", required=True); p.add_argument("--video")
    p = sub.add_parser("reverse"); p.set_defaults(fn=cmd_reverse)
    p.add_argument("--detections", required=True); p.add_argument("--out", required=True)
    p.add_argument("--video")
    p = sub.add_parser("link"); p.set_defaults(fn=cmd_link)
    p.add_argument("--detections", required=True); p.add_argument("--linker", choices=linkers.BUILTIN, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fb"); p.set_defaults(fn=cmd_fb)
    p.add_argument("--detections", required=True); p.add_argument("--linker", choices=linkers.BUILTIN)
    p.add_argument("--forward"); p.add_argument("--backward")
    p = sub.add_parser("consensus"); p.set_defaults(fn=cmd_consensus)
    p.add_argument("--detections", required=True); p.add_argument("--out", required=True)
    p.add_argument("--builtin", default="relax,nn,lap", help="comma list of CBTV-free linkers ('' for none)")
    p.add_argument("--k", type=int, default=1, help="decimation-consensus step (offsets 0..k-1); 1 = off")
    p.add_argument("--no-fb", action="store_true"); p.add_argument("--require-confirm", action="store_true")
    p.add_argument("--member", action="append", help="NAME[:PRODUCER]=TRACKS[@SEQ] full-rate external run")
    p.add_argument("--member-rev", action="append", help="NAME=TRACKS@SEQ run on the reversed sequence")
    p.add_argument("--member-dec", action="append", help="NAME=TRACKS@SEQ run on one decimated offset")
    p = sub.add_parser("score"); p.set_defaults(fn=cmd_score)
    p.add_argument("--reference", required=True); p.add_argument("--tracker", required=True)
    p.add_argument("--tracks", action="append", required=True, help="TRACKS[@SEQ]; repeat for every offset")
    p.add_argument("--gate", type=float, default=None, help="match gate px (default 0.5 d)")
    p = sub.add_parser("audit-sample"); p.set_defaults(fn=cmd_audit_sample)
    p.add_argument("--reference", required=True); p.add_argument("--detections", required=True)
    p.add_argument("--design", choices=["a2", "legacy"], default="a2",
                   help="a2 = Amendment 2 subsample (default); legacy = stratified reference + every disagreement")
    p.add_argument("--n-random", type=int, default=100, help="a2: random agreed links per window")
    p.add_argument("--n-dis", type=int, default=300, help="a2: stratified disagreements per window")
    p.add_argument("--b-frac", type=float, default=0.25, help="a2: share of items annotator B also labels")
    p.add_argument("--n", type=int, default=300, help="legacy: reference links to sample (all disagreements added)")
    p.add_argument("--out", required=True); p.add_argument("--video")
    p.add_argument("--seed", type=int, default=None, help="default: 20261005 (a2) / 0 (legacy)")
    p.add_argument("--lane-bins", type=int, default=3); p.add_argument("--phase-bins", type=int, default=3)
    p.add_argument("--density-bins", type=int, default=3)
    p.add_argument("--frame-offset", type=int, default=0, help="video frame of detection frame 0")
    p = sub.add_parser("audit-report"); p.set_defaults(fn=cmd_audit_report)
    p.add_argument("--audit", required=True); p.add_argument("--a"); p.add_argument("--b")
    p.add_argument("--smallest-effect", type=float, default=None,
                   help="legacy: smallest tracker error rate to resolve (acceptance: upper CI <= this / 5)")
    p.add_argument("--adjudication", help="a2: filled adjudication csv (item_id,...,adjudicated)")
    p.add_argument("--n-boot", type=int, default=2000, help="a2: stratified bootstrap resamples")
    p = sub.add_parser("audit-adjudicate"); p.set_defaults(fn=cmd_audit_adjudicate)
    p.add_argument("--audit", required=True); p.add_argument("--a"); p.add_argument("--b"); p.add_argument("--out")
    p = sub.add_parser("audit-apply"); p.set_defaults(fn=cmd_audit_apply)
    p.add_argument("--reference", required=True); p.add_argument("--detections", required=True)
    p.add_argument("--audit", required=True); p.add_argument("--out", required=True)
    p.add_argument("--a"); p.add_argument("--b")
    p = sub.add_parser("validate"); p.set_defaults(fn=cmd_validate)
    p.add_argument("--out", required=True); p.add_argument("--cbtv", action="store_true", help="also run CBTV (slow)")
    p.add_argument("--fast", action="store_true", help="fewer cells, shorter synthetic clip")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    rc = a.fn(a)
    return int(rc or 0)
