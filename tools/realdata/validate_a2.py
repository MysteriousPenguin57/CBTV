"""Validate the Amendment 2 audit (sample_a2 + report_a2) on synthetic data, where every link's truth
is known.

The CBTV-free reference is built exactly as validate.evaluate does. Simulated annotators A = B = ground truth. The
"full audit" labels every candidate link; the Amendment 2 design labels its subsample. For each dataset:
  - full-audit values: consensus error of the reference links, true links among the disagreements, coverage;
  - the design's estimates with their 95% CIs at the pre-registered seed (does the full value lie inside?);
  - over R further seeds, how often each CI covers the full value (nominal 0.95).

Usage: python -m tools.realdata.validate_a2 [--reps 100] [--frames 240] [--out FILE]"""
import argparse

import numpy as np
import pandas as pd

from . import audit, consensus
from .linkers import diameter
from .validate import det_gt_ids, highfps_data, synth_io_data


def reference_links(d, k_ref=4):
    src = {"path": "synthetic", "sha256": ""}
    members = consensus.builtin_members(d, ("relax", "nn", "lap"), k=k_ref, fb=True, det_source=src)
    return consensus.build(d, members, k=k_ref, det_source=src).links


def full_truth(links, gid):
    ok = (gid[links["det_a"].to_numpy()] >= 0) & (gid[links["det_a"].to_numpy()] == gid[links["det_b"].to_numpy()])
    links = links.assign(truth=ok)
    ref, dis = links[links["status"] == "reference"], links[links["status"] == "disagreement"]
    right = int(ref["truth"].sum())
    true_dis = int(dis["truth"].sum())
    return links, {"consensus_error": 1 - right / len(ref), "dis_true": true_dis,
                   "coverage": right / (right + true_dis) if right + true_dis else float("nan")}


def run_design(links, d, seed, n_boot):
    key, st = audit.sample_a2(links, d, seed=seed)
    ann = pd.DataFrame({"item_id": key["item_id"],             # the key rows carry each link's truth
                        "judgement": np.where(key["truth"].to_numpy(bool), "same", "different")})
    return audit.report_a2(key, st, ann, ann[ann["item_id"].isin(key.loc[key["in_b"], "item_id"])],
                           n_boot=n_boot, seed=seed), key


def validate(name, d, gt, reps, n_boot):
    links = reference_links(d)
    gid = det_gt_ids(d, gt, 0.5 * diameter(d))
    links, full = full_truth(links, gid)
    rep, key = run_design(links, d, audit.A2_SEED, n_boot)
    rows = [{"dataset": name, "seed": "20261005 (pre-registered)", "reference_links": rep["reference_links"],
             "disagreement_links": rep["disagreement_links"], "items": rep["items"],
             "full_consensus_error": full["consensus_error"], "est_consensus_error": rep["consensus_error"],
             "cp95": rep["consensus_error_cp95"], "full_dis_true": full["dis_true"],
             "est_dis_true": rep["dis_true_hat"],
             "dis_true_ci95": rep["dis_true_hat_ci95"], "full_coverage": full["coverage"],
             "est_coverage": rep["coverage"], "coverage_ci95": rep["coverage_ci95"]}]
    cover = {"consensus_error": 0, "dis_true": 0, "coverage": 0}
    for r in range(reps):
        rp, _ = run_design(links, d, audit.A2_SEED + 1 + r, n_boot)
        cover["consensus_error"] += (rp["consensus_error_cp95"][0] <= full["consensus_error"]
                                     <= rp["consensus_error_cp95"][1])
        cover["dis_true"] += rp["dis_true_hat_ci95"][0] <= full["dis_true"] <= rp["dis_true_hat_ci95"][1]
        cover["coverage"] += rp["coverage_ci95"][0] <= full["coverage"] <= rp["coverage_ci95"][1]
    rows[0].update({f"ci_cover_{k}": v / reps if reps else float("nan") for k, v in cover.items()}, reps=reps)
    return rows[0]


def fmt_ci(c):
    return "[%.4g, %.4g]" % tuple(c)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    hd, hgt, _, _ = highfps_data(seed=0, frames=a.frames)
    sd, sgt = synth_io_data()
    res = [validate("synthetic_highfps", hd, hgt, a.reps, a.n_boot), validate("synth_io", sd, sgt, a.reps, a.n_boot)]
    lines = ["| dataset | ref links / disagreements / items | consensus error: full / est (CP 95%%) | true "
             "disagreement "
             "links: full / est (95%%) | coverage: full / est (95%%) | CI covers full value over %d seeds (err / dis / "
             "cov) |"
             % a.reps, "|---|---|---|---|---|---|"]
    for r in res:
        lines.append("| %s | %d / %d / %d | %.4f / %.4f %s | %d / %.1f %s | %.4f / %.4f %s | %.2f / %.2f / %.2f |" % (
            r["dataset"], r["reference_links"], r["disagreement_links"], r["items"], r["full_consensus_error"],
            r["est_consensus_error"], fmt_ci(r["cp95"]), r["full_dis_true"], r["est_dis_true"],
            fmt_ci(r["dis_true_ci95"]), r["full_coverage"], r["est_coverage"], fmt_ci(r["coverage_ci95"]),
            r["ci_cover_consensus_error"], r["ci_cover_dis_true"], r["ci_cover_coverage"]))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return res


if __name__ == "__main__":
    main()
