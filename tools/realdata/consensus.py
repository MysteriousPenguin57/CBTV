"""Consensus pseudo-GT across members and decimation offsets, with provenance and the circularity guard.

Members are link sets on the same detections (``linkers``): builtin CBTV-free linkers, or external tracker runs
snapped onto the detections. A full-rate link a->b becomes a *reference link* iff
  1. every member links a->b at full rate (unanimity; disagreements are never resolved by majority vote),
  2. every member that has a backward pass reproduces it (forward-backward check, ``fb``), and
  3. decimation consensus (step k >= 2): for every start detection p, the k-chain of step-1 candidates
     p -> ... -> q is compared with each member's step-k link of p, taken from its run on the decimated sequence
     whose offset contains p (the k offsets 0..k-1 together give every detection one step-k link). A member that
     links p to another detection, or links another detection to q, *contradicts* the chain and all k links in it
     are rejected. ``require_confirm`` additionally rejects links that no decimated run confirmed.
Every other link proposed by any member is a *disagreement*; it goes to the blinded audit, not to the reference.

Provenance (reference.meta.json) lists each member, its producer, whether it is CBTV-derived, the steps it
contributed and the sha256 of every input. ``check_scoring`` refuses to score
  - anything against a reference without valid provenance or whose files do not match their recorded sha256,
  - CBTV (by name or by output fingerprint) against a CBTV-derived reference,
  - a file that is itself one of the reference's inputs,
  - a member at a step it contributed to the reference.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .fb import builtin_pair, consistency
from .linkers import diameter, link_rows, link_sequence, tracks_to_succ
from .tracksio import read_manifest, read_tracks, remap_frames, sha256_file, write_tracks

SCHEMA = "cbtv-realdata-reference/1"
_CBTV = re.compile(r"cbtv", re.I)


class CircularityError(RuntimeError):
    """Raised when scoring would be circular or the reference provenance cannot be verified."""


@dataclass
class Member:
    name: str
    producer: str
    succ: np.ndarray
    succ_rev: np.ndarray | None = None
    succ_k: np.ndarray | None = None
    cbtv_derived: bool = False
    sources: list = field(default_factory=list)


def is_cbtv_name(s: str) -> bool:
    return bool(_CBTV.search(s or ""))


def looks_like_cbtv(tracks_path) -> list:
    """Output fingerprints of a CBTV run (empty list = none found). Conservative: used only to refuse."""
    p = Path(tracks_path)
    why = []
    if p.with_name(p.stem + "_meta.csv").exists():
        why.append("cbtv _meta.csv sidecar")
    try:
        t = pd.read_csv(p, usecols=lambda c: c in ("missed",))
        if "missed" in t.columns and (t["missed"] > 0).any():
            why.append("coasting rows (missed > 0)")
    except (ValueError, OSError):
        pass
    meta = p.parent / "reference.meta.json"
    if meta.exists():
        try:
            if json.loads(meta.read_text()).get("cbtv_derived"):
                why.append("a CBTV-derived reference")
        except ValueError:
            why.append("unreadable reference meta next to it")
    return why


def builtin_members(d: pd.DataFrame, names=("relax", "nn", "lap"), k: int = 1, fb: bool = True,
                    params: dict | None = None, det_source: dict | None = None) -> list:
    """CBTV-free members on the full-rate sequence (+ backward pass, + decimated offsets 0..k-1)."""
    from .decimate import decimate, frame_range, grid
    params = params or {}
    f0, f1 = frame_range(d)
    full = list(range(f0, f1 + 1))
    out = []
    for nm in names:
        p = params.get(nm)
        if fb:
            succ, succ_rev = builtin_pair(nm, d, full, p)
        else:
            succ, succ_rev = link_sequence(nm, d.assign(frame=d["frame"] - f0), len(full), len(d), p), None
        succ_k = None
        if k >= 2:
            succ_k = np.full(len(d), -1, int)
            for o in range(k):
                dd, src = decimate(d, k, o)
                assert src == grid(f0, f1, k, o)
                s = link_sequence(nm, dd, len(src), len(d), p)
                m = s >= 0
                succ_k[m] = s[m]
        out.append(Member(nm, "builtin:" + nm, succ, succ_rev, succ_k, False,
                          [dict(det_source or {}, role="detections")]))
    return out


def external_member(name: str, producer: str, d: pd.DataFrame, fwd, rev=None, dec=(), k: int = 1,
                    snap_tol: float | None = None) -> Member:
    """Member from tracker output. fwd/rev/dec are (tracks_csv, seq_csv or None) pairs; seq_csv's manifest maps
    processed frames to source frames (None = the full-rate sequence starting at the first detection frame)."""
    tol = snap_tol if snap_tol is not None else 0.4 * diameter(d)
    f0 = int(d["frame"].min())
    n_full = int(d["frame"].max()) - f0 + 1
    sources = []
    why_all: list = []

    def load_run(tracks_csv, seq_csv, role):
        if seq_csv is None:
            src = list(range(f0, f0 + n_full))
        else:
            src = read_manifest(seq_csv)["src_frames"]
        t = remap_frames(read_tracks(tracks_csv), src)
        why_all.extend(looks_like_cbtv(tracks_csv))
        sources.append({"path": str(tracks_csv), "sha256": sha256_file(tracks_csv), "role": role})
        return t, src

    t, src = load_run(fwd[0], fwd[1], "forward")
    succ, _ = tracks_to_succ(t, d, src, tol)
    succ_rev = None
    if rev is not None:
        t, src = load_run(rev[0], rev[1], "backward")
        succ_rev, _ = tracks_to_succ(t, d, src, tol)
    succ_k = None
    if dec:
        succ_k = np.full(len(d), -1, int)
        for pair in dec:
            info = read_manifest(pair[1])
            if int(info["k"]) != k:
                raise ValueError("%s: decimated run has k=%s, consensus step is %d" % (pair[1], info["k"], k))
            t, src = load_run(pair[0], pair[1], "decimated k%d o%d" % (k, info["offset"]))
            s, _ = tracks_to_succ(t, d, src, tol)
            m = s >= 0
            succ_k[m] = s[m]
    cbtv = is_cbtv_name(name) or is_cbtv_name(producer) or bool(why_all)
    return Member(name, producer, succ, succ_rev, succ_k, cbtv, sources)


@dataclass
class Reference:
    accepted: np.ndarray            # succ of reference links
    links: pd.DataFrame             # every candidate link with its status
    meta: dict


def build(d: pd.DataFrame, members: list, k: int = 1, require_confirm: bool = False,
          det_source: dict | None = None, params: dict | None = None) -> Reference:
    if not members:
        raise ValueError("need at least one member")
    N = len(d)
    M = len(members)
    S = np.stack([m.succ for m in members])                     # (M, N)
    first = S[0]
    agree = (first >= 0) & np.all(S == first[None, :], axis=0)
    fb_fail = np.zeros(N, bool)
    n_fb = 0
    for m in members:
        if m.succ_rev is not None:
            n_fb += 1
            fb_fail |= agree & ~consistency(m.succ, m.succ_rev)
    prelim = agree & ~fb_fail
    pre = np.where(prelim, first, -1)

    conflict = np.zeros(N, int)
    confirm = np.zeros(N, int)
    dec_members = [m for m in members if m.succ_k is not None]
    if k >= 2 and dec_members:
        chain = [np.arange(N)]                                  # chain[s][p] = det after s steps from p
        for _ in range(k):
            c = chain[-1]
            chain.append(np.where(c >= 0, pre[np.maximum(c, 0)], -1))
        q = chain[k]
        full = q >= 0
        for m in dec_members:
            sk = m.succ_k
            pred_k = np.full(N, -1, int)
            ok = sk >= 0
            pred_k[sk[ok]] = np.nonzero(ok)[0]
            n_pred = np.bincount(sk[ok], minlength=N)
            p_idx = np.nonzero(full)[0]
            qq = q[p_idx]
            conf = (sk[p_idx] == qq) & (n_pred[qq] == 1)
            contra = ((sk[p_idx] >= 0) & (sk[p_idx] != qq)) | (n_pred[qq] > 1) | \
                     ((pred_k[qq] >= 0) & (pred_k[qq] != p_idx))
            for s in range(k):                                  # mark the k step-1 links of each window
                starts = chain[s][p_idx]
                np.add.at(confirm, starts[conf], 1)
                np.add.at(conflict, starts[contra], 1)
    accepted = prelim & (conflict == 0)
    if require_confirm and k >= 2 and dec_members:
        accepted &= confirm > 0
    acc = np.where(accepted, first, -1)

    # candidate table: every (a, b) any member proposed
    a_all = np.concatenate([np.nonzero(m.succ >= 0)[0] for m in members])
    b_all = np.concatenate([m.succ[m.succ >= 0] for m in members])
    key = np.unique(a_all.astype(np.int64) * N + b_all)
    ca, cb = (key // N).astype(int), (key % N).astype(int)
    prop = np.stack([m.succ[ca] == cb for m in members])        # (M, C)
    n_prop = prop.sum(axis=0)
    is_ref = acc[ca] == cb
    names = [m.name for m in members]
    reasons = []
    for i in range(len(ca)):
        if is_ref[i]:
            reasons.append("")
            continue
        r = []
        if n_prop[i] < M:
            r.append("members %d/%d" % (n_prop[i], M))
        elif fb_fail[ca[i]]:
            r.append("fb")
        elif conflict[ca[i]]:
            r.append("decim conflict")
        elif require_confirm:
            r.append("decim unconfirmed")
        reasons.append(";".join(r))
    fx = d["frame"].to_numpy()
    xy = d[["x", "y"]].to_numpy(float)
    links = pd.DataFrame({
        "frame_a": fx[ca], "det_a": ca, "x_a": xy[ca, 0], "y_a": xy[ca, 1],
        "frame_b": fx[cb], "det_b": cb, "x_b": xy[cb, 0], "y_b": xy[cb, 1],
        "status": np.where(is_ref, "reference", "disagreement"),
        "n_members": n_prop, "fb_fail": fb_fail[ca].astype(int),
        "decim_confirm": confirm[ca], "decim_conflict": conflict[ca], "reasons": reasons,
        "proposers": ["|".join(n for n, pr in zip(names, prop[:, i], strict=True) if pr) for i in range(len(ca))],
    })
    steps = [1] + ([k] if k >= 2 else [])
    meta = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "detections": det_source or {},
        "consensus_step": int(k), "require_confirm": bool(require_confirm),
        "members": [{"name": m.name, "producer": m.producer, "cbtv_derived": bool(m.cbtv_derived),
                     "forward_backward": m.succ_rev is not None,
                     "steps": [1] + ([k] if (k >= 2 and m.succ_k is not None) else []),
                     "sources": m.sources} for m in members],
        "cbtv_derived": any(m.cbtv_derived for m in members),
        "steps": steps,
        "params": params or {},
        "counts": {"detections": int(N), "candidate_links": int(len(ca)), "reference_links": int(accepted.sum()),
                   "disagreements": int((~is_ref).sum()), "unanimous": int(agree.sum()),
                   "fb_fail": int(fb_fail.sum()), "decim_conflict": int((prelim & (conflict > 0)).sum()),
                   "decim_confirmed": int((accepted & (confirm > 0)).sum()), "members_with_fb": n_fb},
    }
    return Reference(acc, links, meta)


def write(ref: Reference, d: pd.DataFrame, out_dir) -> Path:
    """out_dir/reference_tracks.csv (frame,track_id,x,y,r,missed at source frames), reference_links.csv, and
    reference.meta.json holding provenance + the sha256 of both csvs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    on = np.zeros(len(d), bool)
    a = np.nonzero(ref.accepted >= 0)[0]
    on[a] = True
    on[ref.accepted[a]] = True
    dd = d[on]
    tracks = link_rows(ref.accepted, dd, sorted(dd["frame"].unique()))
    write_tracks(out / "reference_tracks.csv", tracks)
    ref.links.to_csv(out / "reference_links.csv", index=False, float_format="%.3f", lineterminator="\n")
    meta = dict(ref.meta)
    meta["files"] = {n: sha256_file(out / n) for n in ("reference_tracks.csv", "reference_links.csv")}
    meta["counts"] = dict(meta["counts"], reference_tracks=int(tracks["track_id"].nunique()) if len(tracks) else 0)
    (out / "reference.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return out


def load(ref_dir):
    """(reference tracks, links, meta) after verifying provenance. Raises CircularityError when unverifiable."""
    ref_dir = Path(ref_dir)
    mp = ref_dir / "reference.meta.json"
    if not mp.exists():
        raise CircularityError("%s has no reference.meta.json: provenance unknown, refusing to score" % ref_dir)
    try:
        meta = json.loads(mp.read_text())
    except ValueError as e:
        raise CircularityError("unreadable %s (%s): refusing to score" % (mp, e)) from e
    if meta.get("schema") != SCHEMA or "members" not in meta or "files" not in meta:
        raise CircularityError("%s is not a %s file: refusing to score" % (mp, SCHEMA))
    for n, h in meta["files"].items():
        p = ref_dir / n
        if not p.exists() or sha256_file(p) != h:
            raise CircularityError("%s does not match the sha256 recorded in its provenance: refusing to score" % p)
    if bool(meta.get("cbtv_derived")) != any(m.get("cbtv_derived") for m in meta["members"]):
        raise CircularityError("%s: cbtv_derived flag inconsistent with its members: refusing to score" % mp)
    return read_tracks(ref_dir / "reference_tracks.csv"), pd.read_csv(ref_dir / "reference_links.csv"), meta


def check_scoring(meta: dict, tracker: str, step: int, files=()) -> None:
    """Raise CircularityError if scoring `tracker` (its output `files`) at `step` against this reference would be
    circular. The tracker counts as CBTV by name or by any CBTV fingerprint of its output."""
    fingerprints = {str(f): looks_like_cbtv(f) for f in files}
    scored_cbtv = is_cbtv_name(tracker) or any(fingerprints.values())
    if scored_cbtv and meta.get("cbtv_derived"):
        bad = [m["name"] for m in meta["members"] if m.get("cbtv_derived")]
        why = "named %r" % tracker if is_cbtv_name(tracker) else "output fingerprint %s" % (
            sorted({w for v in fingerprints.values() for w in v}))
        raise CircularityError("refusing to score CBTV (%s) against a reference built from CBTV output "
                               "(members %s); build the reference without CBTV" % (why, bad))
    shas = {s.get("sha256") for m in meta["members"] for s in m.get("sources", []) if s.get("sha256")}
    for f in files:
        if sha256_file(f) in shas:
            raise CircularityError("refusing to score %s: it is one of the reference's own inputs" % f)
    for m in meta["members"]:
        if m["name"].lower() == tracker.lower() and int(step) in m.get("steps", []):
            raise CircularityError("refusing to score %s at step %d: it is a reference member at that step"
                                   % (tracker, step))
