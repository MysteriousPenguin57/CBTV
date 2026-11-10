"""Cascade Tier 1 — exclusion / competition pruning (Phase 4 Stage 4).

The cascade thesis (IMPROVEMENT_PLAN ◆): certainty can be reached through
STRUCTURE instead of measurement. Tier 1 implements the exclusion half on
the fall-through population (tracks no easy/constellation commit settled):
a near-certain (track -> detection) pair is committed and its detection is
DELETED from every competitor's candidate row; deletions can promote a
competitor's second choice to near-certainty, so the pass iterates to a
fixpoint ("if X->Y is near-certain and A is unlikely->Y, then A->B by
elimination — no further work on A").

Relationship to the shipped stages: `greedy_confident` ([STAGE-1]) is the
single-pass special case. Tier 1 generalizes it with target deletion +
iteration, shrinking what reaches the 2x2/Hungarian stages.

Guardrail (dispatch, binding): commits here use the SAME margin/max-cost
bars as stage 1 — nothing is pruned below a measured alternative; a pair
is only ever committed when its margin over the row's second-best clears
`margin` and its cost clears `max_cost`. With `--cascade-tier1 0`
(default) nothing here runs — byte-identical baseline.

Net-compute accounting (the ship criterion): Tier 1 pays one row-min scan
per iteration over the residual matrix; it saves Hungarian work on the
committed rows/cols. Both sides are measured in the Stage-4 cells.
"""
from __future__ import annotations

import numpy as np


def tier1_exclusion_pass(
    C: np.ndarray,
    rem_tr: list,
    rem_dt: list,
    *,
    margin: float,
    max_cost: float,
    max_iters: int = 3,
):
    """Iterative near-certain commit + target exclusion.

    Parameters
    ----------
    C : (M, N) cost matrix (full-index space).
    rem_tr, rem_dt : remaining (unmatched) track/det indices.
    margin, max_cost : stage-1 confidence bars (args.conf_margin /
        args.conf_maxcost) — identical semantics to greedy_confident.
    max_iters : exclusion-chain depth (3 covers ~all fixpoints in
        practice; each extra iteration only sees newly-promoted rows).

    Returns
    -------
    pairs : list of (track_idx, det_idx) committed near-certainly.
    n_iters_used : int — iterations that produced at least one commit.
    """
    if not rem_tr or not rem_dt:
        return [], 0

    sub = C[np.ix_(rem_tr, rem_dt)].astype(np.float64, copy=True)
    tr = np.asarray(rem_tr)
    dt = np.asarray(rem_dt)
    row_alive = np.ones(len(tr), dtype=bool)
    col_alive = np.ones(len(dt), dtype=bool)
    pairs = []
    n_used = 0

    for _ in range(max_iters):
        if not row_alive.any() or not col_alive.any():
            break
        work = sub[np.ix_(row_alive, col_alive)]
        if work.shape[1] == 0:
            break
        r_idx = np.where(row_alive)[0]
        c_idx = np.where(col_alive)[0]

        best_j = np.argmin(work, axis=1)
        best_v = work[np.arange(work.shape[0]), best_j]
        if work.shape[1] > 1:
            w2 = work.copy()
            w2[np.arange(work.shape[0]), best_j] = np.inf
            second_v = w2.min(axis=1)
        else:
            second_v = np.full(work.shape[0], np.inf)

        confident = (best_v < max_cost) & ((second_v - best_v) >= margin)
        if not confident.any():
            break

        # One-to-one within the iteration: if two confident rows want the
        # same column, only the cheaper one commits this round (the other
        # re-evaluates next iteration once the column is gone).
        order = np.argsort(best_v, kind="stable")
        claimed = set()
        committed_this_round = False
        for k in order:
            if not confident[k]:
                continue
            j = int(best_j[k])
            if j in claimed:
                continue
            claimed.add(j)
            gi = int(tr[r_idx[k]])
            gj = int(dt[c_idx[j]])
            pairs.append((gi, gj))
            row_alive[r_idx[k]] = False
            col_alive[c_idx[j]] = False
            committed_this_round = True
        if not committed_this_round:
            break
        n_used += 1

    return pairs, n_used
