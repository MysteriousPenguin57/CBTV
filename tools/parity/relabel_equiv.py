"""Are two tracks.csv equal up to a track-id relabelling? (C004 tolerance check)

Usage: python tools/parity/relabel_equiv.py A.csv B.csv
Checks (1) the multiset of (frame, x, y, r, missed) rows is identical, ignoring track_id, and (2) every track of A
maps to exactly one track of B over all its rows (a bijection on ids). Exit 0 iff both hold. ASCII output only."""
import sys

import pandas as pd

KEY = ["frame", "x", "y", "r", "missed"]


def main(a_path, b_path):
    a, b = pd.read_csv(a_path, dtype=str), pd.read_csv(b_path, dtype=str)
    rows_equal = len(a) == len(b) and a[KEY].sort_values(KEY).values.tolist() == b[KEY].sort_values(KEY).values.tolist()
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    a_to_b = m.groupby("track_id_a")["track_id_b"].nunique()
    b_to_a = m.groupby("track_id_b")["track_id_a"].nunique()
    bijection = (len(m) == len(a) and (a_to_b == 1).all() and (b_to_a == 1).all()
                 and len(a_to_b) == a.track_id.nunique() == b.track_id.nunique())
    relabelled = int((m.track_id_a != m.track_id_b).groupby(m.track_id_a).any().sum())
    print("rows A/B %d/%d  rows identical ignoring track_id: %s  id bijection: %s  tracks A/B %d/%d  "
          "tracks with a different id: %d" % (len(a), len(b), rows_equal, bool(bijection), a.track_id.nunique(),
                                               b.track_id.nunique(), relabelled))
    return 0 if rows_equal and bijection else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
