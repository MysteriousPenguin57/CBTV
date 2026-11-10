"""T032 / B15: METRIC deltas of a CPU run against the committed goldens, per fixture x preset (not only "bytes differ").

Usage: python tools/ci/platform_metrics.py <rundir> [--run] [--summary FILE] [--csv FILE]
<rundir> has the layout tools/ci/parity_report.py and tools/run_fixtures.py write:
<rundir>/<preset>/<fixture>/tracks.csv.
--run first runs `tools/run_fixtures.py cpu <rundir>/<preset> --preset <preset>` for both presets.
Per fixture x preset: track count, mean track length (rows per track) and coasting % (rows with missed > 0) for the run
and the golden; rows of the run matched to the golden by nearest position in the same frame (track ids ignored):
fraction within 0.5 / 1 / 5 px and the max distance. HOTA/AssA of synth_io against its gt.csv needs an
evaluator that is not shipped here: reported as "not computed". Prints a Markdown table (and appends it to
--summary, e.g. $GITHUB_STEP_SUMMARY); --csv writes the numbers. Exit 0 unless a run is missing. ASCII only."""
import argparse
import csv
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ["primary60", "m05_30", "synth_io"]
GOLDEN = {"ship-v4": "golden_gpu_tracks.csv", "bt": "golden_gpu_tracks_bt.csv"}


def load(p):
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame"]), int(r["track_id"]), float(r["x"]), float(r["y"]), int(float(r["missed"]))))
    return rows


def summary(rows):
    ids = {r[1] for r in rows}
    return dict(tracks=len(ids), mean_len=len(rows) / max(len(ids), 1),
                coast_pct=100.0 * sum(r[4] > 0 for r in rows) / max(len(rows), 1))


def position_match(got, gold):
    """Per frame, each run row's distance to the nearest golden row (ids ignored)."""
    import numpy as np
    from scipy.spatial import cKDTree

    def by_frame(rows):
        out = {}
        for fr, _, x, y, _ in rows:
            out.setdefault(fr, []).append((x, y))
        return {k: np.array(v) for k, v in out.items()}
    a, b = by_frame(got), by_frame(gold)
    d = [cKDTree(b[fr]).query(a[fr])[0] for fr in sorted(a) if fr in b]
    unmatched = sum(len(a[fr]) for fr in a if fr not in b)
    d = np.concatenate(d) if d else np.zeros(0)
    n = len(d) + unmatched
    def frac(t):
        return (d <= t).sum() / n if n else float("nan")
    return dict(within_0_5=frac(0.5), within_1=frac(1.0), within_5=frac(5.0),
                max_d=float(d.max()) if len(d) else float("nan"), rows_no_golden_frame=unmatched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary")
    ap.add_argument("--csv")
    a = ap.parse_args()
    out_rows, missing = [], 0
    for preset, golden in GOLDEN.items():
        out = Path(a.rundir) / preset
        if a.run:
            r = subprocess.run([sys.executable, "tools/run_fixtures.py", "cpu", str(out), "--preset", preset],
                               cwd=str(ROOT), capture_output=True, text=True)
            print(r.stdout + r.stderr, flush=True)
        for fx in FIXTURES:
            got_p, gold_p = out / fx / "tracks.csv", ROOT / "tests" / "fixtures" / fx / golden
            row = dict(preset=preset, fixture=fx)
            if not got_p.exists():
                missing += 1
                row["note"] = "run missing"
                out_rows.append(row)
                continue
            got, gold = load(got_p), load(gold_p)
            sg, sb = summary(got), summary(gold)
            row.update({f"run_{k}": v for k, v in sg.items()})
            row.update({f"golden_{k}": v for k, v in sb.items()})
            row.update(position_match(got, gold))
            row["byte_identical"] = got_p.read_bytes() == gold_p.read_bytes()
            row["hota"] = "not computed" if fx == "synth_io" else "n/a (no GT in fixture)"
            out_rows.append(row)
    head = "%s %s / Python %s" % (platform.system(), platform.machine(), platform.python_version())
    t = ["### Platform metrics vs goldens: " + head, "",
         "| preset | fixture | byte-identical | tracks run/golden | mean length run/golden | coasting % run/golden "
         "| rows <=0.5 / <=1 / <=5 px | max dpos px | HOTA |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        if "note" in r:
            t.append("| %s | %s | %s | | | | | | |" % (r["preset"], r["fixture"], r["note"]))
            continue
        t.append("| %s | %s | %s | %d / %d | %.1f / %.1f | %.2f / %.2f | %.4f / %.4f / %.4f | %.3f | %s |" % (
            r["preset"], r["fixture"], "yes" if r["byte_identical"] else "no", r["run_tracks"], r["golden_tracks"],
            r["run_mean_len"], r["golden_mean_len"], r["run_coast_pct"], r["golden_coast_pct"],
            r["within_0_5"], r["within_1"], r["within_5"], r["max_d"], r["hota"]))
    text = "\n".join(t) + "\n"
    print(text)
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as f:
            f.write(text)
    if a.csv:
        keys = sorted({k for r in out_rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(out_rows)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
