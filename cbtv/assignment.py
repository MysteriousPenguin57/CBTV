"""3-stage assignment: confident greedy -> 2x2 -> Hungarian."""
import numpy as np

# Required: without it Stage 3 is skipped and tracks.csv changes.
from scipy.optimize import linear_sum_assignment as hungarian


def compute_kernel_confidence(R, sNCC):
    """
    Analyze correlation map to estimate match confidence

    Returns confidence (0-1) based on:
    - Uniqueness: one clear peak vs multiple similar peaks
    - Signal-to-noise: how much peak stands out
    - Sharpness: sharp peak vs broad blob
    """
    peak_val = float(R.max())
    peak_y, peak_x = np.unravel_index(R.argmax(), R.shape)

    # Uniqueness: mask peak region and find second best
    R_masked = R.copy()
    mask_radius = 5
    y0 = max(0, peak_y - mask_radius)
    y1 = min(R.shape[0], peak_y + mask_radius + 1)
    x0 = max(0, peak_x - mask_radius)
    x1 = min(R.shape[1], peak_x + mask_radius + 1)
    R_masked[y0:y1, x0:x1] = -999

    if R_masked.size > 0 and R_masked.max() > -999:
        second_peak = float(R_masked.max())
        uniqueness = (peak_val - second_peak) / (peak_val + 1e-6)
        uniqueness = np.clip(uniqueness, 0.0, 1.0)
    else:
        uniqueness = 1.0

    # Signal-to-noise
    mean_bg = float(R.mean())
    std_bg = float(R.std()) + 1e-6
    snr = (peak_val - mean_bg) / std_bg
    snr_normalized = np.clip(snr / 5.0, 0.0, 1.0)

    # Sharpness
    if (peak_y > 0 and peak_y < R.shape[0]-1 and
        peak_x > 0 and peak_x < R.shape[1]-1):
        local_patch = R[peak_y-1:peak_y+2, peak_x-1:peak_x+2]
        sharpness = float(peak_val - local_patch.mean()) / (local_patch.std() + 1e-6)
        sharpness_normalized = np.clip(sharpness / 3.0, 0.0, 1.0)
    else:
        sharpness_normalized = 0.5

    confidence = (
        0.4 * uniqueness +
        0.3 * snr_normalized +
        0.2 * sharpness_normalized +
        0.1 * peak_val
    )

    return float(np.clip(confidence, 0.0, 1.0))

def greedy_confident(C, margin, max_cost):
    # === [STAGE-1] Greedy with margin test (see Flowchart 16 — Stage 1) ===
    # [Decision Rationale] The margin test (v2 - v >= margin) is what makes
    # this safe to run before the Hungarian: a track is only locked early if
    # its best detection is meaningfully better than its second-best. The
    # max_cost gate (default ~25 px) prevents extremely poor matches from
    # being locked even when they look "confident" by margin alone.
    # [v6 Change] In v6 the input matrix C is much smaller because PKT
    # (Phase 4C) has already locked most tracks before this function runs —
    # typical M ≈ 4-10 instead of M ≈ 540 in the original.
    """Stage 1: Match tracks with high confidence"""
    M,N = C.shape
    matched_tr=set(); matched_dt=set(); pairs=[]
    for i in range(M):
        row=C[i]; j=int(np.argmin(row)); v=float(row[j])
        if v>=max_cost: continue
        v2 = float(np.partition(row,1)[1]) if N>1 else 1e9
        if (v2 - v) >= margin and j not in matched_dt:
            pairs.append((i,j)); matched_tr.add(i); matched_dt.add(j)
    return pairs, matched_tr, matched_dt

def greedy_confident_sparse(C, margin, max_cost):
    # [Phase5-speed lever 1] Byte-identical vectorisation of greedy_confident.
    # The shipped loop runs np.argmin + np.partition per row (M Python iters);
    # this computes both order-statistics for every row in 3 vectorised numpy
    # calls, then resolves detection conflicts sequentially in ascending row
    # order -- exactly the original's commit order, so the first row to claim a
    # detection wins identically. Asserted byte-identical in _smoke_sparse_gate.
    M, N = C.shape
    if M == 0 or N == 0:
        return [], set(), set()
    J = C.argmin(axis=1)                       # per-row argmin (first-min, == np.argmin)
    V = C[np.arange(M), J]
    if N > 1:
        V2 = np.partition(C, 1, axis=1)[:, 1]  # per-row 2nd order statistic
    else:
        V2 = np.full(M, 1e9)
    ok = (V < max_cost) & ((V2 - V) >= margin)
    matched_tr = set(); matched_dt = set(); pairs = []
    for i in np.nonzero(ok)[0]:                 # ascending -> first claim wins (as the loop)
        j = int(J[i])
        if j not in matched_dt:
            i = int(i)
            pairs.append((i, j)); matched_tr.add(i); matched_dt.add(j)
    return pairs, matched_tr, matched_dt


def solve_2x2_blocks_sparse(C, remaining_tr, remaining_dt, max_cost=40.0):
    # [Phase5-speed lever 1] STRICTLY byte-identical fast path for the #1 t_asn
    # hotspot (solve_2x2_blocks, ~84% of t_asn). The ONLY change vs the shipped
    # loop is O(1) set membership instead of O(len) `in list` scans -- the
    # measured bulk (~1.6M C-level int comparisons/frame; ~27->5 ms, 5.4x). The
    # per-row/col np.argsort(C[i]) / np.argsort(C[:,j]) calls are IDENTICAL to
    # the baseline, so the (cost-affecting, cascade-sensitive) argsort tie order
    # matches bit-for-bit -- no divergence even on exact-cost ties. A whole-
    # matrix np.argsort precompute was rejected: axis=1/0 argsort does NOT
    # reproduce 1D tie order, and 2x2's commit cascade turns those tie
    # differences into cost-CHANGING divergences (CBTV_ASSIGN_PARITY: 8 real
    # mismatches, costΔ up to -60). Asserted exact in _smoke_sparse_gate.
    pairs = []; used_tr = set(); used_dt = set()
    rem_tr = list(remaining_tr); rem_dt = list(remaining_dt)
    rts = set(rem_tr); rds = set(rem_dt)
    for i in rem_tr:
        if i in used_tr: continue
        cols = np.argsort(C[i], kind="stable")[:2]
        for j in cols:
            j = int(j)
            if j not in rds or j in used_dt: continue
            cand_tracks = np.argsort(C[:, j], kind="stable")[:2]
            for k in cand_tracks:
                k = int(k)
                if k == i or k not in rts or k in used_tr: continue
                cols_k = [int(c) for c in np.argsort(C[k], kind="stable")[:2] if int(c) in rds and int(c) != j]
                if not cols_k: continue
                j2 = cols_k[0]
                a = C[i, j] + C[k, j2]
                b = C[i, j2] + C[k, j]
                if min(a, b) >= max_cost * 2: continue
                if a <= b:
                    if j not in used_dt and j2 not in used_dt:
                        pairs += [(i, j), (k, j2)]
                        used_tr.update([i, k]); used_dt.update([j, j2])
                else:
                    if j not in used_dt and j2 not in used_dt:
                        pairs += [(i, j2), (k, j)]
                        used_tr.update([i, k]); used_dt.update([j, j2])
                break
    return pairs


def solve_2x2_blocks(C, remaining_tr, remaining_dt, max_cost=40.0):
    # === [STAGE-2] 2×2 permutation enumeration (see Flowchart 16 — Stage 2) ===
    # [Decision Rationale] The 2×2 stage handles "swap-prone" pairs: when
    # tracks i and k are each contesting the same two detections j and j2,
    # we enumerate the two possible permutations and pick the lower total
    # cost. This is cheaper and more deterministic than letting the
    # Hungarian solver resolve them, and it avoids the cascade swaps that
    # cause IDF1 regressions when many local pairs are jointly assigned.
    # The "min(a,b) >= max_cost*2" gate (≈ 80 px) discards permutations
    # whose total is too poor to trust.
    """Stage 2: Resolve ambiguous pairs using 2×2 optimization"""
    pairs=[]; used_tr=set(); used_dt=set()
    rem_tr=list(remaining_tr); rem_dt=list(remaining_dt)
    for i in rem_tr:
        if i in used_tr: continue
        cols = np.argsort(C[i], kind="stable")[:2]
        for j in cols:
            if j not in rem_dt or j in used_dt: continue
            cand_tracks = np.argsort(C[:,j], kind="stable")[:2]
            for k in cand_tracks:
                if k==i or k not in rem_tr or k in used_tr: continue
                cols_k = [c for c in np.argsort(C[k], kind="stable")[:2] if c in rem_dt and c!=j]
                if not cols_k: continue
                j2 = cols_k[0]
                a = C[i,j] + C[k,j2]
                b = C[i,j2] + C[k,j]
                if min(a,b) >= max_cost*2: continue
                if a<=b:
                    if j not in used_dt and j2 not in used_dt:
                        pairs += [(i,j),(k,j2)]
                        used_tr.update([i,k]); used_dt.update([j,j2])
                else:
                    if j not in used_dt and j2 not in used_dt:
                        pairs += [(i,j2),(k,j)]
                        used_tr.update([i,k]); used_dt.update([j,j2])
                break
    return pairs

def solve_hungarian_stage3(C, remaining_tr, remaining_dt, max_cost=120.0):
    # === [STAGE-3] Hungarian on residual (see Flowchart 16 — Stage 3) ===
    # [Decision Rationale] Hungarian is the most expensive solver and the
    # least deterministic when costs are close. By the time we reach this
    # stage, only ambiguous tracks/detections that survived Stages 1-2
    # remain (typically <10 of each). The max_cost=120 gate (≈4× the
    # Stage 1 gate) is intentionally generous: at this point we'd rather
    # accept a marginal match than leave a track unassigned, because
    # spawn/prune treats unmatched tracks harshly.
    #
    # [Phase5-speed] A sentinel-prune of all-sentinel rows/cols was prototyped
    # to shrink this O(n^3) solve, but (a) at PRIMARY scale the scipy LSA is
    # <0.5% of t_asn -- the real hotspot is solve_2x2_blocks (~84%, sped up
    # separately under --sparse-gate) -- and (b) the prune is NOT byte-identical
    # when rem_tr > rem_dt (the frozen case): dropping all-sentinel rows changes
    # which finite row wins a contested column in scipy's incomplete matching
    # (CBTV_ASSIGN_PARITY: cost-changing divergence). So Stage 3 stays baseline.
    """Stage 3: Hungarian on all remaining tracks/detections"""
    if not remaining_tr or not remaining_dt or hungarian is None:
        return []

    rem_tr = list(remaining_tr)
    rem_dt = list(remaining_dt)
    C_sub = C[np.ix_(rem_tr, rem_dt)]

    row_idx, col_idx = hungarian(C_sub)
    pairs = []
    for ri, ci in zip(row_idx, col_idx, strict=False):
        if C_sub[ri, ci] < max_cost:
            pairs.append((rem_tr[ri], rem_dt[ci]))

    return pairs

