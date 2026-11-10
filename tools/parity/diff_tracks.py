"""Compare two tracks.csv files (CPU vs GPU, new vs golden). Generalises tools/golden/diff_tracks.py.

Usage: python tools/parity/diff_tracks.py A.csv B.csv
Prints rows, whether the files are byte-identical, the (frame, track_id) pairs shared / only-A /
only-B, the max position delta on shared pairs, track-id counts, and the first differing frame
(first frame where the sets of (track_id, x, y, r, missed) rows differ). Exit 0 iff byte-identical.
ASCII output only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(a_path, b_path):
    """Return a dict of stats; keys documented in the module docstring."""
    ident = Path(a_path).read_bytes() == Path(b_path).read_bytes()
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    out = {"rows_A": len(a), "rows_B": len(b), "byte_identical": ident,
           "ids_A": int(a["track_id"].nunique()), "ids_B": int(b["track_id"].nunique())}
    m = a.merge(b, on=["frame", "track_id"], how="outer", suffixes=("_A", "_B"), indicator=True)
    both = m[m["_merge"] == "both"]
    out["shared"] = len(both)
    out["only_A"] = int((m["_merge"] == "left_only").sum())
    out["only_B"] = int((m["_merge"] == "right_only").sum())
    out["max_dpos"] = float(np.hypot(both["x_A"] - both["x_B"], both["y_A"] - both["y_B"]).max()) if len(both) else float("nan")
    out["first_diff_frame"] = None
    if not ident:
        cols = [c for c in a.columns if c in b.columns]
        ka = a[cols].round(6).astype(str).agg("|".join, axis=1)
        kb = b[cols].round(6).astype(str).agg("|".join, axis=1)
        diff_frames = set(a.loc[~ka.isin(set(kb)), "frame"]) | set(b.loc[~kb.isin(set(ka)), "frame"])
        out["first_diff_frame"] = int(min(diff_frames)) if diff_frames else None
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__); return 2
    s = compare(sys.argv[1], sys.argv[2])
    print("rows A/B: %d/%d  byte-identical: %s" % (s["rows_A"], s["rows_B"], s["byte_identical"]))
    print("(frame,id) shared/only-A/only-B: %d/%d/%d  max |dpos| on shared: %.3f  ids A/B: %d/%d" % (
        s["shared"], s["only_A"], s["only_B"], s["max_dpos"], s["ids_A"], s["ids_B"]))
    if not s["byte_identical"]:
        print("first differing frame: %s" % s["first_diff_frame"])
    return 0 if s["byte_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
