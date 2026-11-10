"""Compare two CBTV_DUMP_C directories (C_<frame:05d>.npz from T006), CPU vs GPU.

Usage: python tools/parity/diff_dumps.py DIR_A DIR_B [--summary summary.csv] [--atol 0.0]
Per frame: n_pairs_A/B (valid_pairs.sum()), n_valid_diff (entries whose valid bit differs),
n_finite_diff (finite-vs-inf/nan mismatches), n_c_diff (finite entries differing by > atol),
max_abs_dC (over entries finite in both). Prints the first differing frame and, in it, the first
(track_id, det index) pair that differs with both values and both valid_pairs bits.
Exit 0 when every frame is identical (and both directories hold the same frames), else 1.
ASCII output only."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def frames(d):
    return {int(p.stem.split("_")[1]): p for p in Path(d).glob("C_*.npz")}


def compare_frame(a, b, atol=0.0):
    """Return (row dict, first_diff dict or None) for two loaded npz mappings."""
    row = {"n_pairs_A": int(a["valid_pairs"].sum()), "n_pairs_B": int(b["valid_pairs"].sum()),
           "n_valid_diff": -1, "n_finite_diff": -1, "n_c_diff": -1, "max_abs_dC": float("nan")}
    first = None
    if a["C"].shape != b["C"].shape or not np.array_equal(a["track_ids"], b["track_ids"]) \
            or not np.array_equal(a["det_idx"], b["det_idx"]):
        row["max_abs_dC"] = float("nan")
        first = {"kind": "shape/ids", "A": "C%s tracks=%d dets=%d" % (a["C"].shape, len(a["track_ids"]), len(a["det_idx"])),
                 "B": "C%s tracks=%d dets=%d" % (b["C"].shape, len(b["track_ids"]), len(b["det_idx"]))}
        return row, first
    Ca, Cb = a["C"], b["C"]
    va, vb = a["valid_pairs"], b["valid_pairs"]
    fa, fb = np.isfinite(Ca), np.isfinite(Cb)
    both = fa & fb
    dC = np.zeros(Ca.shape)
    dC[both] = np.abs(Ca[both] - Cb[both])
    vdiff, fdiff, cdiff = va != vb, fa != fb, both & (dC > atol)
    row.update(n_valid_diff=int(vdiff.sum()), n_finite_diff=int(fdiff.sum()), n_c_diff=int(cdiff.sum()),
               max_abs_dC=float(dC[both].max()) if both.any() else 0.0)
    bad = vdiff | fdiff | cdiff
    if bad.any():
        # row-major first differing entry
        i, j = (int(v) for v in np.argwhere(bad)[0])
        first = {"kind": "values", "track_id": int(a["track_ids"][i]), "det_idx": int(a["det_idx"][j]),
                 "C_A": float(Ca[i, j]), "C_B": float(Cb[i, j]),
                 "valid_A": bool(va[i, j]), "valid_B": bool(vb[i, j])}
    return row, first


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--summary", default=None, help="write per-frame summary CSV here")
    ap.add_argument("--atol", type=float, default=0.0)
    a = ap.parse_args()
    fa, fb = frames(a.dir_a), frames(a.dir_b)
    if not fa or not fb:
        print("no C_*.npz in %s" % (a.dir_a if not fa else a.dir_b)); return 2
    rows, first_frame, first_info = [], None, None
    for f in sorted(set(fa) | set(fb)):
        if f not in fa or f not in fb:
            rows.append({"frame": f, "n_pairs_A": "", "n_pairs_B": "", "n_valid_diff": "", "n_finite_diff": "",
                         "n_c_diff": "", "max_abs_dC": "", "note": "only in " + ("A" if f in fa else "B")})
            if first_frame is None:
                first_frame, first_info = f, {"kind": "missing", "where": "only in " + ("A" if f in fa else "B")}
            continue
        with np.load(fa[f]) as za, np.load(fb[f]) as zb:
            row, first = compare_frame(za, zb, a.atol)
        row = {"frame": f, **row, "note": ""}
        rows.append(row)
        if first is not None and first_frame is None:
            first_frame, first_info = f, first
    if a.summary:
        with open(a.summary, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["frame", "n_pairs_A", "n_pairs_B", "n_valid_diff", "n_finite_diff",
                                               "n_c_diff", "max_abs_dC", "note"])
            w.writeheader(); w.writerows(rows)
    n_bad = sum(1 for r in rows if r["note"] or r["n_valid_diff"] not in (0,) or r["n_finite_diff"] != 0 or r["n_c_diff"] != 0)
    print("frames A/B: %d/%d   differing frames: %d" % (len(fa), len(fb), n_bad))
    if first_frame is None:
        print("IDENTICAL"); return 0
    r = next(x for x in rows if x["frame"] == first_frame)
    print("first differing frame: %d (max|dC| over frame = %s)" % (first_frame, r["max_abs_dC"]))
    if first_info["kind"] == "values":
        print("  first pair: track_id=%d det_idx=%d  C_A=%r C_B=%r  valid_A=%s valid_B=%s" % (
            first_info["track_id"], first_info["det_idx"], first_info["C_A"], first_info["C_B"],
            first_info["valid_A"], first_info["valid_B"]))
    else:
        print("  " + "; ".join("%s=%s" % kv for kv in first_info.items()))
    return 1


if __name__ == "__main__":
    sys.exit(main())
