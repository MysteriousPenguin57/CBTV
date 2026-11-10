"""Relaxation-PTV linker (Baek & Lee 1996 / Ohmi & Li 2000; Okamoto spring variant as a flag).

A baseline of the paper (a stand-in for Jiang 2018). numpy/scipy only.

Per frame pair A=t, B=t+1 with d = 2 x median detection radius:
  cand[i] = B points within R_s of A_i (+ same colour bin if colour_mode == "gate")
  nbr[i]  = other A points within R_n of A_i
  P over cand[i] + {j0 (death)}: init uniform, P[i,j0] = P0, rows normalised; then up to N_it times
    Q[i,j]   = sum_{k in nbr[i]} sum_{l in cand[k], |(xB_l-xB_j)-(xA_k-xA_i)| <= R_c} P[k,l] * wc(k,l)
               (spring: weight exp(-|dv|^2 / (2 R_c^2)) instead of the hard test)
    P'[i,j]  = P[i,j] * (A + B*Q[i,j]) * wc(i,j);   P'[i,j0] = P[i,j0] * (A + B*Q_floor);   normalise
  stop when max|P' - P| < eps. Accept (i, argmax_j P[i,:]) iff P >= P_acc and the pair is a mutual argmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


@dataclass
class RelaxParams:
    R_n: float = 3.0          # neighbour radius, units of d
    R_c: float = 0.35         # displacement-consistency tolerance, units of d
    N_it: int = 10            # iterations (capped at 20)
    eps: float = 1e-3
    A: float = 0.3
    B: float = 3.0
    P_acc: float = 0.5
    P0: float = 0.1
    Q_floor: float = 0.0      # support credited to the death hypothesis (0: decays like an unsupported match)
    R_s_mult: float = 1.0     # multiplies the default (or given) search radius
    spring: bool = False
    colour_mode: str = "none"  # none | gate | weight
    w_off: float = 0.05       # colour weight for differing bins (mode weight)
    max_pairs: int = 20000
    memory: int = 0           # frames an unmatched track may coast before it dies


def _mutual_nn(a, b):
    """Index pairs (i, j) where a_i and b_j are each other's nearest neighbour."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    _, ja = cKDTree(b).query(a)
    _, ib = cKDTree(a).query(b)
    i = np.arange(len(a))
    keep = ib[ja] == i
    return i[keep], ja[keep]


def displacement_q95(frames_xy):
    """q95 of |displacement| from a global-shift + mutual-NN pass over consecutive frames (detections only)."""
    mags = []
    for a, b in zip(frames_xy[:-1], frames_xy[1:], strict=False):
        i, j = _mutual_nn(a, b)
        if len(i) == 0:
            continue
        shift = np.median(b[j] - a[i], axis=0)
        i, j = _mutual_nn(a + shift, b)
        mags.append(np.linalg.norm(b[j] - a[i], axis=1))
    if not mags:
        return None
    return float(np.percentile(np.concatenate(mags), 95))


def default_search_radius(d, q95):
    """R_s = clip(max(1.1d, 1.5 q95), 1.1d, 4.5d)."""
    if q95 is None:
        return 1.1 * d
    return float(np.clip(max(1.1 * d, 1.5 * q95), 1.1 * d, 4.5 * d))


def link_pair(xyA, xyB, d, R_s, params=None, colA=None, colB=None, log=None):
    """Link A (n,2) to B (m,2). Returns (matches [(i, j, P)], info dict)."""
    p = RelaxParams() if params is None else params
    xyA = np.asarray(xyA, float).reshape(-1, 2)
    xyB = np.asarray(xyB, float).reshape(-1, 2)
    nA, nB = len(xyA), len(xyB)
    info = {"R_s": R_s, "pairs": 0, "iters": 0}
    if nA == 0 or nB == 0:
        return [], info
    colour = p.colour_mode != "none"
    if colour and (colA is None or colB is None):
        raise ValueError("colour_mode %r needs colour bins for both frames" % p.colour_mode)
    treeB = cKDTree(xyB)
    while True:
        lists = treeB.query_ball_point(xyA, R_s)
        ia = np.repeat(np.arange(nA), [len(c) for c in lists])
        jb = np.fromiter((j for c in lists for j in c), int, count=len(ia))
        if p.colour_mode == "gate":
            keep = np.asarray(colA)[ia] == np.asarray(colB)[jb]
            ia, jb = ia[keep], jb[keep]
        if len(ia) <= p.max_pairs:
            break
        R_s *= 0.9
        if log:
            log("[RELAX] %d candidate pairs > %d: R_s -> %.2f" % (len(ia), p.max_pairs, R_s))
    info["R_s"], info["pairs"] = R_s, len(ia)
    if len(ia) == 0:
        return [], info
    order = np.lexsort((jb, ia))
    ia, jb = ia[order], jb[order]
    disp = xyB[jb] - xyA[ia]
    if p.colour_mode == "weight":
        wc = np.where(np.asarray(colA)[ia] == np.asarray(colB)[jb], 1.0, p.w_off)
    else:
        wc = np.ones(len(ia))

    # Support matrix S (pairs x pairs): S[(i,j),(k,l)] = consistency weight for k in nbr[i].
    R_n, R_c = p.R_n * d, p.R_c * d
    starts = np.searchsorted(ia, np.arange(nA + 1))
    nbrs = cKDTree(xyA).query_ball_point(xyA, R_n)
    rows, cols, vals = [], [], []
    cut = 3.0 * R_c if p.spring else R_c
    for i in range(nA):
        pi = np.arange(starts[i], starts[i + 1])
        if len(pi) == 0:
            continue
        pk = np.concatenate([np.arange(starts[k], starts[k + 1]) for k in nbrs[i] if k != i] or [np.zeros(0, int)])
        if len(pk) == 0:
            continue
        dv = np.linalg.norm(disp[pi][:, None, :] - disp[pk][None, :, :], axis=2)
        r, c = np.nonzero(dv <= cut)
        if len(r) == 0:
            continue
        rows.append(pi[r]); cols.append(pk[c])
        vals.append(np.exp(-dv[r, c] ** 2 / (2 * R_c ** 2)) if p.spring else np.ones(len(r)))
    C = len(ia)
    if rows:
        S = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(C, C))
    else:
        S = sparse.csr_matrix((C, C))

    # P init: uniform over cand[i], P0 for death, rows normalised.
    ncand = np.bincount(ia, minlength=nA).astype(float)
    P = 1.0 / ncand[ia] / (1.0 + p.P0)
    P_death = np.where(ncand > 0, p.P0 / (1.0 + p.P0), 1.0)
    death_gain = p.A + p.B * p.Q_floor
    for it in range(min(int(p.N_it), 20)):
        Q = S @ (P * wc)
        Pn = P * (p.A + p.B * Q) * wc
        Dn = P_death * death_gain
        tot = np.bincount(ia, weights=Pn, minlength=nA) + Dn
        Pn = Pn / tot[ia]
        Dn = Dn / tot
        delta = max(np.abs(Pn - P).max(), np.abs(Dn - P_death).max())
        P, P_death = Pn, Dn
        info["iters"] = it + 1
        if delta < p.eps:
            break

    # Accept mutual argmax with P >= P_acc (and above the death probability).
    best_i = np.full(nA, -1)
    best_p = np.full(nA, -1.0)
    np.maximum.at(best_p, ia, P)
    is_best_i = P == best_p[ia]
    best_for_j = np.full(nB, -1.0)
    np.maximum.at(best_for_j, jb, P)
    matches = []
    for c in np.nonzero(is_best_i)[0]:
        i, j = ia[c], jb[c]
        if best_i[i] >= 0:      # tie: keep the first
            continue
        best_i[i] = c
        if P[c] >= p.P_acc and P[c] > P_death[i] and P[c] == best_for_j[j]:
            matches.append((int(i), int(j), float(P[c])))
    seen = set()
    out = []
    for m in matches:            # a tied B column keeps only its first A
        if m[1] not in seen:
            seen.add(m[1]); out.append(m)
    return out, info


def track(frames, params=None, log=None):
    """Link a sequence. frames: list of (frame_id, xy (n,2), r (n,), colour (n,) or None).

    d = 2 x median radius over the live tracks and frame t+1. Unmatched tracks coast for `memory` frames
    (linked from their last position), then die. Returns rows (frame_id, track_id, x, y, r, missed=0), info.
    """
    p = RelaxParams() if params is None else params
    q95 = displacement_q95([np.asarray(f[1], float).reshape(-1, 2) for f in frames])
    rows = []
    next_id = 1
    live = []   # [track_id, x, y, r, colour, frames_unmatched]

    def spawn(fid, x, y, r, c, tid=None):
        nonlocal next_id
        if tid is None:
            tid, next_id = next_id, next_id + 1
        rows.append((fid, tid, float(x), float(y), float(r), 0))
        return [tid, float(x), float(y), float(r), c, 0]

    if frames:
        fid, xy, r, col = frames[0]
        live = [spawn(fid, xy[n][0], xy[n][1], r[n], None if col is None else col[n]) for n in range(len(xy))]
    for fid, xy, r, col in frames[1:]:
        xy = np.asarray(xy, float).reshape(-1, 2)
        r = np.asarray(r, float)
        radii = np.concatenate([[t[3] for t in live], r])
        if len(radii) == 0:
            continue
        d = 2.0 * float(np.median(radii))
        R_s = default_search_radius(d, q95) * p.R_s_mult
        xyA = np.array([[t[1], t[2]] for t in live], float).reshape(-1, 2)
        colA = None if col is None else np.array([t[4] for t in live])
        matches, _ = link_pair(xyA, xy, d, R_s, p, colA, None if col is None else np.asarray(col), log)
        matched_A = {i for i, _, _ in matches}
        tid_of_B = {j: live[i][0] for i, j, _ in matches}
        new_live = []
        for i, t in enumerate(live):
            if i not in matched_A and t[5] < p.memory:
                t[5] += 1
                new_live.append(t)
        for j in range(len(xy)):
            new_live.append(spawn(fid, xy[j][0], xy[j][1], r[j], None if col is None else col[j], tid_of_B.get(j)))
        live = new_live
    return rows, {"q95": q95, "n_tracks": next_id - 1}
