"""Diff new vs published tracks.csv. Usage: diff_tracks.py new.csv pub.csv"""
import sys

import numpy as np
import pandas as pd

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
print("rows new/pub", len(a), len(b), "cols same", list(a.columns) == list(b.columns))
if len(a) == len(b) and list(a.columns) == list(b.columns):
    ident = a.equals(b)
    print("identical", ident)
    if not ident:
        ne = (a.fillna(-9e9) != b.fillna(-9e9)).any(axis=1)
        i = int(np.argmax(ne.values)); print("first differing row", i); print(a.iloc[i].to_dict()); print(b.iloc[i].to_dict())
        for c in ("x", "y"):
            if c in a: print("max|d%s|" % c, float((a[c] - b[c]).abs().max()))
