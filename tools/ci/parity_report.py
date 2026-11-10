"""C014: per fixture x preset, compare a CPU run against its golden: byte-identical / ids-only / different.

Usage: python tools/ci/parity_report.py <outdir> [--summary FILE]
Runs tools/run_fixtures.py cpu for both presets (ship-v4 -> golden_gpu_tracks.csv, bt -> golden_gpu_tracks_bt.csv),
then cmp + tools/parity/relabel_equiv.py; for a real difference, also the first differing frame
(tools/parity/diff_tracks.py). Prints a Markdown table (and appends it to --summary, e.g. $GITHUB_STEP_SUMMARY).
Exit 0 iff everything is byte-identical. ASCII only."""
import argparse
import csv
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ["primary60", "m05_30", "synth_io"]
GOLDEN = {"ship-v4": "golden_gpu_tracks.csv", "bt": "golden_gpu_tracks_bt.csv"}


def drift(got, gold):
    """Nearest-position match per frame (ignoring track ids): how far apart are the two runs' rows?"""
    import numpy as np
    from scipy.spatial import cKDTree

    def load(p):
        by = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                by.setdefault(int(r["frame"]), []).append((float(r["x"]), float(r["y"]), float(r["r"])))
        return {k: np.array(v) for k, v in by.items()}
    a, b = load(got), load(gold)
    d, dr, first = [], [], None
    for fr in sorted(set(a) | set(b)):
        if fr not in a or fr not in b:
            continue
        dist, j = cKDTree(b[fr][:, :2]).query(a[fr][:, :2])
        d.append(dist)
        dr.append(np.abs(a[fr][:, 2] - b[fr][j, 2]))
        if first is None and (dist.max() > 0 or dr[-1].max() > 0):
            first = fr
    d, dr = np.concatenate(d), np.concatenate(dr)
    return ("rows matched by position: exact %d, <=0.01 px %d, <=0.5 px %d, >0.5 px %d of %d; max |dxy| %.3f px, "
            "max |dr| %.3f px; first frame with any drift %s"
            % ((d == 0).sum(), (d <= 0.01).sum(), (d <= 0.5).sum(), (d > 0.5).sum(), len(d), d.max(), dr.max(), first))


def run(cmd):
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--summary")
    a = ap.parse_args()
    rows, all_ok = [], True
    for preset, golden in GOLDEN.items():
        out = Path(a.outdir) / preset
        r = run([sys.executable, "tools/run_fixtures.py", "cpu", str(out), "--preset", preset])
        print(r.stdout + r.stderr, flush=True)
        for fx in FIXTURES:
            got, gold = out / fx / "tracks.csv", ROOT / "tests" / "fixtures" / fx / golden
            if not got.exists():
                res, all_ok = "tracker failed (see run.log)", False
            elif got.read_bytes() == gold.read_bytes():
                res = "byte-identical"
            else:
                all_ok = False
                eq = run([sys.executable, "tools/parity/relabel_equiv.py", str(got), str(gold)])
                if eq.returncode == 0:
                    n = eq.stdout.strip().split("tracks with a different id: ")[-1]
                    res = "ids-only (%s tracks renumbered)" % n
                else:
                    d = run([sys.executable, "tools/parity/diff_tracks.py", str(got), str(gold)])
                    res = "different: " + " ".join(d.stdout.split())[:300] + " | " + drift(got, gold)
            rows.append((preset, fx, res))
    head = "%s %s / Python %s" % (platform.system(), platform.machine(), platform.python_version())
    table = ["### Fixture parity: " + head, "", "| preset | fixture | result |", "|---|---|---|"]
    table += ["| %s | %s | %s |" % r for r in rows]
    text = "\n".join(table) + "\n"
    print(text)
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as f:
            f.write(text)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
