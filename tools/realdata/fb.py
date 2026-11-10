"""Forward-backward identity check.

Track forward, then on the reversed sequence; a forward link a->b is consistent iff the backward pass links b->a.
Works on any link set: builtin linkers run both passes here, an external tracker supplies its run on the reversed
sequence (``python -m tools.realdata reverse`` writes the reversed detections + clip + manifest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decimate import derive
from .linkers import link_sequence


def consistency(succ_f: np.ndarray, succ_b: np.ndarray) -> np.ndarray:
    """Bool over det_id: forward link exists and the backward link set maps its target back to it."""
    ok = np.zeros(len(succ_f), bool)
    a = np.nonzero(succ_f >= 0)[0]
    ok[a] = succ_b[succ_f[a]] == a
    return ok


def rate(succ_f: np.ndarray, succ_b: np.ndarray) -> float:
    """Fraction of forward links that the backward pass reproduces (nan when there are no forward links)."""
    n = int((succ_f >= 0).sum())
    return float(consistency(succ_f, succ_b).sum() / n) if n else float("nan")


def builtin_pair(name: str, d: pd.DataFrame, src_frames, params: dict | None = None):
    """(succ_forward, succ_backward) of a builtin linker on the source frames src_frames (and their reverse)."""
    src = [int(f) for f in src_frames]
    fwd = link_sequence(name, derive(d, src), len(src), len(d), params)
    rev = src[::-1]
    bwd = link_sequence(name, derive(d, rev), len(rev), len(d), params)
    return fwd, bwd
