"""Blinded audit of a reference: stratified sampler, crop export, Cohen's kappa, reference error rate, merge.

sample   a random stratified sample of reference links (strata: lane = x bin, phase = frame bin, local density
         tertile) plus every disagreement link, shuffled under opaque random item ids. Written:
           audit_key.csv            sealed: item -> link, kind (reference / disagreement), stratum, proposers
           audit_strata.csv         stratum population and sample size (for the stratified estimate)
           annotator_A.csv, annotator_B.csv   item_id,image,judgement,comment (independent orders; nothing else)
           crops/<item>.png         raw pixels only: left = frame t window centred on detection a, right = the same
                                    window in frame t+g; the only marks are ticks in a grey margin outside the image,
                                    at a's position (left) and b's position (right). No overlay, no tracker hint.
           INSTRUCTIONS.txt
report   Cohen's kappa of the two filled csvs; reference error rate (both annotators "different") with a Wilson
         95% CI, a conservative rate counting disputed items as errors, a stratified point estimate, and the
         acceptance test (upper CI <= smallest tracker error to resolve / 5).
apply    a new reference: drops reference links both annotators judged different, adds disagreement links both
         judged same (conflicting additions are dropped); provenance records the audit files' sha256.

Amendment 2 (the design the CLI uses by default):
sample_a2   per window 100 simple-random agreed links + 300 disagreements by stratified random sampling (strata:
            proposing members x pair-distance tertile in d x frame third; proportional, >= 10 per non-empty stratum),
            seed 20261005. The sealed key stores each item's inclusion probability `pi` and HT weight `w`;
            annotator A gets every item, annotator B a seeded random 25% (`in_b`), in an independent order.
report_a2   consensus error of the random agreed links with a Clopper-Pearson 95% CI; Horvitz-Thompson estimates of
            the true links among the disagreements and the reference coverage; 95% CIs by stratified bootstrap
            (2000 resamples); Cohen's kappa on the A/B overlap. A-B disagreements use the adjudicated label
            (adjudication_sheet writes the CSV to fill); unresolved/unsure items are counted and left out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import consensus
from .linkers import diameter
from .tracksio import sha256_file

LABELS = ("same", "different", "unsure")
_ALIAS = {"s": "same", "same": "same", "1": "same", "y": "same", "yes": "same",
          "d": "different", "different": "different", "diff": "different", "0": "different", "n": "different",
          "no": "different", "u": "unsure", "unsure": "unsure", "?": "unsure"}
INSTRUCTIONS = """Blinded link audit. Work alone; do not compare notes with the other annotator.
Each image shows two crops of the same window of the video: left = frame t, right = a later frame.
The grey ticks in the margins of the left crop point at one particle. The ticks in the margins of the right crop
point at a position in the later frame. Question: is the particle at the right-hand ticks the SAME physical particle
as the one at the left-hand ticks?
Fill the 'judgement' column of your csv with: same / different / unsure. Optional free text in 'comment'.
"""


def strata(links: pd.DataFrame, d: pd.DataFrame, lane_bins: int = 3, phase_bins: int = 3,
           density_bins: int = 3) -> pd.DataFrame:
    """lane (x bin), phase (frame bin), density (quantile bin of detections within 3 d of a) per link."""
    out = links.copy()
    x0, x1 = float(d["x"].min()), float(d["x"].max()) + 1e-9
    f0, f1 = float(d["frame"].min()), float(d["frame"].max()) + 1e-9
    out["lane"] = np.clip(((out["x_a"] - x0) / (x1 - x0) * lane_bins).astype(int), 0, lane_bins - 1)
    out["phase"] = np.clip(((out["frame_a"] - f0) / (f1 - f0) * phase_bins).astype(int), 0, phase_bins - 1)
    rad = 3.0 * diameter(d)
    dens = np.zeros(len(out), int)
    by = {int(f): g for f, g in d.groupby("frame", sort=False)}
    for f, g in out.groupby("frame_a", sort=False):
        pts = by[int(f)][["x", "y"]].to_numpy(float)
        tree = cKDTree(pts)
        cnt = tree.query_ball_point(g[["x_a", "y_a"]].to_numpy(float), rad, return_length=True)
        dens[g.index.to_numpy()] = np.asarray(cnt) - 1
    out["n_neigh"] = dens
    if density_bins > 1 and len(out):
        qs = np.quantile(dens, np.linspace(0, 1, density_bins + 1)[1:-1])
        out["density"] = np.searchsorted(qs, dens, side="right")
    else:
        out["density"] = 0
    out["stratum"] = ("L" + out["lane"].astype(str) + "-P" + out["phase"].astype(str) + "-D"
                      + out["density"].astype(str))
    return out


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional allocation, >= 1 per non-empty stratum, never more than the stratum holds."""
    tot = int(sizes.sum())
    if tot == 0:
        return sizes * 0
    a = np.maximum(1, np.round(n * sizes / tot)).astype(int)
    return pd.Series(np.minimum(a, sizes.to_numpy()), index=sizes.index)


def sample(links: pd.DataFrame, d: pd.DataFrame, n_ref: int, seed: int = 0, **bins):
    """(key, strata table). key rows: sampled reference links + all disagreements, shuffled, with item ids."""
    rng = np.random.default_rng(seed)
    s = strata(links.reset_index(drop=True), d, **bins)
    ref = s[s["status"] == "reference"]
    sizes = ref.groupby("stratum").size()
    alloc = allocate(sizes, n_ref)
    picked = []
    for h, g in ref.groupby("stratum", sort=True):
        k = int(alloc.get(h, 0))
        if k:
            picked.append(g.iloc[np.sort(rng.choice(len(g), k, replace=False))])
    dis = s[s["status"] == "disagreement"]
    key = pd.concat(picked + [dis], ignore_index=True)
    key["kind"] = np.where(key["status"] == "reference", "reference", "disagreement")
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    ids: set = set()
    while len(ids) < len(key):
        ids.add("%08x" % int(rng.integers(0, 2 ** 32)))
    key.insert(0, "item_id", [str(x) for x in rng.permutation(np.array(sorted(ids), dtype=object))])
    st = pd.DataFrame({"stratum": sizes.index, "population": sizes.to_numpy(),
                       "sampled": [int(alloc.get(h, 0)) for h in sizes.index]})
    return key, st


# ------------------------------------------------------------------ Amendment 2: design-based subsample
A2_SEED = 20261005


def strata_a2(links: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Amendment 2 strata of the disagreement links: which reference members proposed the link (CBTV is never a
    member), pair-distance tertile in d (tertiles over this window's disagreements), frame third."""
    s = links.reset_index(drop=True).copy()
    s["pair_d"] = np.hypot(s["x_b"] - s["x_a"], s["y_b"] - s["y_a"]) / diameter(d)
    dis = s["status"] == "disagreement"
    if dis.any():
        q = np.quantile(s.loc[dis, "pair_d"], [1 / 3, 2 / 3])
        s["dist_t"] = np.searchsorted(q, s["pair_d"].to_numpy(), side="right")
    else:
        s["dist_t"] = 0
    f0, f1 = float(d["frame"].min()), float(d["frame"].max()) + 1e-9
    s["frame_t"] = np.clip(((s["frame_a"] - f0) / (f1 - f0) * 3).astype(int), 0, 2)
    prop = s["proposers"] if "proposers" in s.columns else pd.Series("", index=s.index)
    s["members"] = prop.fillna("").map(lambda p: "|".join(sorted(x for x in str(p).split("|") if x)) or "none")
    if s["members"].map(consensus.is_cbtv_name).any():
        raise consensus.CircularityError("a CBTV-named member proposed an audited link")
    s["stratum"] = "M:" + s["members"] + "|T" + s["dist_t"].astype(str) + "|F" + s["frame_t"].astype(str)
    return s


def allocate_min(sizes: pd.Series, n: int, n_min: int = 10) -> pd.Series:
    """Proportional allocation of n over the non-empty strata with a minimum of n_min each (all of a stratum when it
    holds fewer), capped at the stratum size; exact total n when feasible (greedy towards the proportional
    targets). If the minimums alone exceed n they are kept (total > n); if n >= the population, everything."""
    sizes = sizes[sizes > 0].sort_index()
    N = int(sizes.sum())
    if n >= N:
        return sizes.copy()
    a = np.minimum(sizes.to_numpy(), n_min).astype(int)
    target = n * sizes.to_numpy() / N
    cap = sizes.to_numpy()
    while a.sum() < n:
        room = a < cap
        gap = np.where(room, target - a, -np.inf)
        a[int(np.argmax(gap))] += 1                    # largest shortfall first; ties -> first stratum (sorted)
    return pd.Series(a, index=sizes.index)


def sample_a2(links: pd.DataFrame, d: pd.DataFrame, n_random: int = 100, n_dis: int = 300, seed: int = A2_SEED,
              b_frac: float = 0.25, n_min: int = 10):
    """(key, strata table) of the Amendment 2 design for one window. key: n_random simple-random agreed links
    (design 'random') + n_dis stratified disagreements, each with its inclusion probability `pi` and HT weight
    `w` = 1/pi, shuffled under opaque item ids; `in_b` marks the seeded b_frac subset annotator B also labels."""
    rng = np.random.default_rng(seed)
    s = strata_a2(links, d)
    ref = s[s["status"] == "reference"]
    k = min(n_random, len(ref))
    r = ref.iloc[np.sort(rng.choice(len(ref), k, replace=False))].copy() if k else ref.iloc[:0].copy()
    r["design"], r["stratum"], r["population"] = "random", "random", len(ref)
    r["pi"] = k / len(ref) if len(ref) else np.nan
    dis = s[s["status"] == "disagreement"]
    sizes = dis.groupby("stratum").size()
    alloc = allocate_min(sizes, n_dis, n_min)
    parts = [r]
    for h, g in dis.groupby("stratum", sort=True):
        m = int(alloc.get(h, 0))
        if m:
            p = g.iloc[np.sort(rng.choice(len(g), m, replace=False))].copy()
            p["design"], p["population"], p["pi"] = "disagreement", len(g), m / len(g)
            parts.append(p)
    key = pd.concat(parts, ignore_index=True)
    key["w"] = 1.0 / key["pi"]
    key["kind"] = np.where(key["status"] == "reference", "reference", "disagreement")
    key = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
    ids: set = set()
    while len(ids) < len(key):
        ids.add("%08x" % int(rng.integers(0, 2 ** 32)))
    key.insert(0, "item_id", [str(x) for x in rng.permutation(np.array(sorted(ids), dtype=object))])
    key["in_b"] = False
    nb = int(round(b_frac * len(key)))
    if nb:
        key.loc[rng.choice(len(key), nb, replace=False), "in_b"] = True
    st = pd.concat([pd.DataFrame({"stratum": ["random"], "design": ["random"], "population": [len(ref)],
                                  "sampled": [k]}),
                    pd.DataFrame({"stratum": sizes.index, "design": "disagreement", "population": sizes.to_numpy(),
                                  "sampled": [int(alloc.get(h, 0)) for h in sizes.index]})], ignore_index=True)
    st["pi"] = st["sampled"] / st["population"].where(st["population"] > 0)
    return key, st


def design_note(st: pd.DataFrame, n_random: int = 100, n_dis: int = 300) -> str:
    """Realised Amendment 2 sample sizes for the packet (DESIGN.txt); no labels, links or tracker names."""
    r = st[st["design"] == "random"]
    dis = st[st["design"] == "disagreement"]
    n_d = int(dis["sampled"].sum())
    lines = ["Amendment 2 audit design.",
             "random agreed links: %d sampled of %d (nominal "
             "%d)" % (int(r["sampled"].sum()), int(r["population"].sum()),
                                                                    n_random),
             "disagreements: %d sampled of %d in %d non-empty strata (nominal %d)" % (
                 n_d, int(dis["population"].sum()), len(dis), n_dis)]
    if n_d > n_dis:
        lines.append("realised disagreements exceed the nominal %d because the minimum of 10 per non-empty stratum "
                     "(or all of a smaller stratum) takes precedence (by design)." % n_dis)
    elif n_d < n_dis:
        lines.append("realised disagreements are fewer than nominal because the window has only %d."
                     % int(dis["population"].sum()))
    return "\n".join(lines) + "\n"


def _frames(video, wanted):
    import cv2
    need = set(int(f) for f in wanted)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError("cannot open %s" % video)
    out, i, last = {}, 0, max(need) if need else -1
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in need:
            out[i] = fr
        i += 1
    cap.release()
    return out


def crop_pair(fa, fb, xa, ya, xb, yb, half: int, scale: int = 4, margin: int = 10):
    """Side-by-side crops with margin ticks only (no pixels of the image are drawn on)."""
    import cv2
    H, W = fa.shape[:2]
    cx, cy = int(round(xa)), int(round(ya))
    x0, y0 = cx - half, cy - half

    def window(img):
        # (2*half+1)^2 window at (x0, y0), zero outside the image: clip to the image, then place at the right offset,
        # so the size is constant and no index wraps, wherever the centre is.
        out = np.zeros((2 * half + 1, 2 * half + 1) + img.shape[2:], img.dtype)
        ys, xs = max(y0, 0), max(x0, 0)
        ye, xe = min(y0 + 2 * half + 1, H), min(x0 + 2 * half + 1, W)
        if ye > ys and xe > xs:
            out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
        return out

    def panel(img, px, py):
        w = cv2.resize(window(img), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        s = w.shape[0]
        p = np.full((s + 2 * margin, s + 2 * margin, 3), 128, np.uint8)
        p[margin:margin + s, margin:margin + s] = w
        u = int(round((px - x0 + 0.5) * scale)) + margin
        v = int(round((py - y0 + 0.5) * scale)) + margin
        tick = (255, 255, 255)
        if margin <= u < margin + s:
            cv2.line(p, (u, 0), (u, margin - 2), tick, 2)
            cv2.line(p, (u, s + margin + 1), (u, s + 2 * margin), tick, 2)
        if margin <= v < margin + s:
            cv2.line(p, (0, v), (margin - 2, v), tick, 2)
            cv2.line(p, (s + margin + 1, v), (s + 2 * margin, v), tick, 2)
        return p

    left, right = panel(fa, xa, ya), panel(fb, xb, yb)
    gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
    return np.hstack([left, gap, right])


def export(key: pd.DataFrame, st: pd.DataFrame, out_dir, video=None, d_px: float = 16.0, seed: int = 0,
           frame_offset: int = 0) -> Path:
    """Write the audit package. frame_offset: video frame index of detection frame 0 (usually 0)."""
    import cv2
    out = Path(out_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    key.to_csv(out / "audit_key.csv", index=False, float_format="%.3f", lineterminator="\n")
    st.to_csv(out / "audit_strata.csv", index=False, lineterminator="\n")
    rng = np.random.default_rng(seed + 1)
    for who in ("A", "B"):
        ids = key["item_id"].to_numpy()
        if who == "B" and "in_b" in key.columns:        # Amendment 2: B labels only the seeded overlap subset
            ids = ids[key["in_b"].to_numpy(bool)]
        order = rng.permutation(len(ids))
        ann = pd.DataFrame({"item_id": ids[order]})
        ann["image"] = "crops/" + ann["item_id"] + ".png"
        ann["judgement"] = ""
        ann["comment"] = ""
        ann.to_csv(out / ("annotator_%s.csv" % who), index=False, lineterminator="\n")
    (out / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS)
    if video is not None and len(key):
        half = int(round(1.5 * d_px))
        fr = _frames(video, set(key["frame_a"] + frame_offset) | set(key["frame_b"] + frame_offset))
        for r in key.itertuples(index=False):
            img = crop_pair(fr[int(r.frame_a) + frame_offset], fr[int(r.frame_b) + frame_offset],
                            r.x_a, r.y_a, r.x_b, r.y_b, half)
            cv2.imwrite(str(out / "crops" / ("%s.png" % r.item_id)), img)
    return out


def normalise(lbl) -> str:
    s = str(lbl).strip().lower() if lbl is not None and not (isinstance(lbl, float) and np.isnan(lbl)) else ""
    return _ALIAS.get(s, "")


def cohen_kappa(a, b) -> float:
    """Cohen's kappa of two label sequences (items with an empty label in either are dropped)."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x and y]
    if not pairs:
        return float("nan")
    cats = sorted({c for p in pairs for c in p})
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n) for c in cats)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def judgements(key: pd.DataFrame, ann_a: pd.DataFrame, ann_b: pd.DataFrame) -> pd.DataFrame:
    a = ann_a.set_index("item_id")["judgement"].map(normalise)
    b = ann_b.set_index("item_id")["judgement"].map(normalise)
    j = key.copy()
    j["a"] = j["item_id"].map(a).fillna("")
    j["b"] = j["item_id"].map(b).fillna("")
    return j


def report(key: pd.DataFrame, st: pd.DataFrame, ann_a: pd.DataFrame, ann_b: pd.DataFrame,
           smallest_effect: float | None = None) -> dict:
    j = judgements(key, ann_a, ann_b)
    both = j[(j["a"] != "") & (j["b"] != "")]
    ref = both[both["kind"] == "reference"]
    err = (ref["a"] == "different") & (ref["b"] == "different")
    disputed = ~err & ((ref["a"] != ref["b"]) | (ref["a"] == "unsure"))
    n = len(ref)
    lo, hi = wilson(int(err.sum()), n)
    clo, chi = wilson(int((err | disputed).sum()), n)
    pop = st.set_index("stratum")["population"]
    W = pop / pop.sum() if pop.sum() else pop
    strat = 0.0
    for h, g in ref.assign(_e=err.to_numpy()).groupby("stratum"):
        strat += float(W.get(h, 0.0)) * float(g["_e"].mean())
    dis = both[both["kind"] == "disagreement"]
    out = {
        "items": int(len(j)), "judged_by_both": int(len(both)),
        "kappa": cohen_kappa(both["a"].tolist(), both["b"].tolist()),
        "raw_agreement": float((both["a"] == both["b"]).mean()) if len(both) else float("nan"),
        "reference_items": n, "reference_errors": int(err.sum()), "reference_disputed": int(disputed.sum()),
        "ref_error_rate": float(err.mean()) if n else float("nan"), "ref_error_ci95": [lo, hi],
        "ref_error_rate_conservative": float((err | disputed).mean()) if n else float("nan"),
        "ref_error_conservative_ci95": [clo, chi], "ref_error_rate_stratified": strat if n else float("nan"),
        "disagreement_items": int(len(dis)),
        "disagreements_same": int(((dis["a"] == "same") & (dis["b"] == "same")).sum()),
        "disagreements_different": int(((dis["a"] == "different") & (dis["b"] == "different")).sum()),
    }
    if smallest_effect is not None:
        out["acceptance_threshold"] = smallest_effect / 5.0
        out["accepted"] = bool(chi <= smallest_effect / 5.0) if n else False
        out["accepted_rule"] = "conservative upper 95% CI <= smallest tracker error / 5"
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact two-sided (1 - alpha) binomial CI."""
    from scipy.stats import beta
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def final_labels(key: pd.DataFrame, ann_a: pd.DataFrame, ann_b: pd.DataFrame,
                 adjudication: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per item: a, b (B only on the overlap) and the label used: A's; on the overlap where A and B disagree, the
    adjudicated label (column `adjudicated`), else unresolved (''). 'unsure' counts as unresolved."""
    j = judgements(key, ann_a, ann_b)
    lab = j["a"].copy()
    lab[(lab == "") & (j["b"] != "")] = j["b"]
    clash = (j["a"] != "") & (j["b"] != "") & (j["a"] != j["b"])
    adj = {}
    if adjudication is not None and len(adjudication):
        adj = adjudication.set_index("item_id")["adjudicated"].map(normalise).to_dict()
    lab[clash] = j.loc[clash, "item_id"].map(adj).fillna("")
    j["label"] = lab.where(lab != "unsure", "")
    return j


def _estimates(rnd_err: np.ndarray, n_ref_pop: int, dis_by_stratum: dict, pops: dict) -> dict:
    """Point estimates from resolved labels: rnd_err = 0/1 per judged random item; dis_by_stratum = {h: 0/1 array of
    'same' (a disagreement that is a true link)}; pops = {h: N_h}. Within a stratum the resolved items stand for its
    whole population (HT with the realised sample size)."""
    p_err = float(rnd_err.mean()) if len(rnd_err) else float("nan")
    wrong = n_ref_pop * p_err
    true_dis = sum(pops[h] * float(v.mean()) for h, v in dis_by_stratum.items() if len(v))
    right = n_ref_pop - wrong
    cov = right / (right + true_dis) if (right + true_dis) > 0 else float("nan")
    return {"ref_error": p_err, "ref_wrong_hat": wrong, "dis_true_hat": true_dis, "coverage": cov}


def report_a2(key: pd.DataFrame, st: pd.DataFrame, ann_a: pd.DataFrame, ann_b: pd.DataFrame,
              adjudication: pd.DataFrame | None = None, n_boot: int = 2000, seed: int = A2_SEED) -> dict:
    """Amendment 2 report for one window (see module doc). Coverage = estimated share of the true links in the
    candidate pool (reference + disagreement links) that the reference holds."""
    j = final_labels(key, ann_a, ann_b, adjudication)
    rnd = j[(j["design"] == "random") & (j["label"] != "")]
    rnd_err = (rnd["label"] == "different").to_numpy(int)
    n_ref_pop = int(st.loc[st["design"] == "random", "population"].sum())
    pops = st[st["design"] == "disagreement"].set_index("stratum")["population"].to_dict()
    dj = j[(j["design"] == "disagreement") & (j["label"] != "")]
    dis_by = {h: (g["label"] == "same").to_numpy(int) for h, g in dj.groupby("stratum")}
    est = _estimates(rnd_err, n_ref_pop, dis_by, pops)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        rb = rng.choice(rnd_err, len(rnd_err), replace=True) if len(rnd_err) else rnd_err
        db = {h: rng.choice(v, len(v), replace=True) for h, v in dis_by.items() if len(v)}
        boot.append(_estimates(rb, n_ref_pop, db, pops))
    bdf = pd.DataFrame(boot)
    ci = lambda c: [float(np.nanpercentile(bdf[c], 2.5)), float(np.nanpercentile(bdf[c], 97.5))]  # noqa: E731
    ov = j[j["in_b"].astype(bool) & (j["a"] != "") & (j["b"] != "")] if "in_b" in j.columns else j.iloc[:0]
    missing_strata = sorted(h for h in pops if h not in dis_by)
    k_err = int(rnd_err.sum())
    return {
        "items": int(len(j)), "random_items": int((j["design"] == "random").sum()),
        "disagreement_items": int((j["design"] == "disagreement").sum()),
        "unresolved": int((j["label"] == "").sum()),
        "overlap_items": int(j["in_b"].astype(bool).sum()) if "in_b" in j.columns else 0,
        "overlap_judged": int(len(ov)), "kappa": cohen_kappa(ov["a"].tolist(), ov["b"].tolist()),
        "overlap_disagreements": int((ov["a"] != ov["b"]).sum()),
        "consensus_errors": k_err, "consensus_judged": int(len(rnd_err)),
        "consensus_error": est["ref_error"], "consensus_error_cp95": list(clopper_pearson(k_err, len(rnd_err))),
        "reference_links": n_ref_pop, "ref_wrong_hat": est["ref_wrong_hat"],
        "disagreement_links": int(sum(pops.values())), "dis_true_hat": est["dis_true_hat"],
        "dis_true_hat_ci95": ci("dis_true_hat"), "coverage": est["coverage"], "coverage_ci95": ci("coverage"),
        "ref_error_boot_ci95": ci("ref_error"), "n_boot": n_boot,
        "strata_without_resolved_items": missing_strata,
    }


def adjudication_sheet(key: pd.DataFrame, ann_a: pd.DataFrame, ann_b: pd.DataFrame, path) -> Path:
    """CSV of the overlap items where A and B disagree, for joint adjudication: item_id,image,a,b,adjudicated."""
    j = judgements(key, ann_a, ann_b)
    c = j[(j["a"] != "") & (j["b"] != "") & (j["a"] != j["b"])]
    out = pd.DataFrame({"item_id": c["item_id"], "image": "crops/" + c["item_id"] + ".png", "a": c["a"], "b": c["b"],
                        "adjudicated": ""})
    out.to_csv(path, index=False, lineterminator="\n")
    return Path(path)


def apply(ref_dir, d: pd.DataFrame, audit_dir, out_dir, ann_a=None, ann_b=None) -> Path:
    """New reference dir with the audit's verdicts applied (see module doc)."""
    ref_dir, audit_dir = Path(ref_dir), Path(audit_dir)
    _, links, meta = consensus.load(ref_dir)
    key = pd.read_csv(audit_dir / "audit_key.csv", dtype={"item_id": str})
    pa = Path(ann_a) if ann_a else audit_dir / "annotator_A.csv"
    pb = Path(ann_b) if ann_b else audit_dir / "annotator_B.csv"
    j = judgements(key, pd.read_csv(pa, dtype={"item_id": str}), pd.read_csv(pb, dtype={"item_id": str}))
    both_diff = j[(j["a"] == "different") & (j["b"] == "different") & (j["kind"] == "reference")]
    both_same = j[(j["a"] == "same") & (j["b"] == "same") & (j["kind"] == "disagreement")]
    acc = np.full(len(d), -1, int)
    r = links[links["status"] == "reference"]
    acc[r["det_a"].to_numpy()] = r["det_b"].to_numpy()
    acc[both_diff["det_a"].to_numpy()] = -1
    add = both_same[["det_a", "det_b"]].to_numpy()
    taken_b = set(acc[acc >= 0].tolist())
    ca = pd.Series(add[:, 0]).value_counts() if len(add) else pd.Series(dtype=int)
    cb = pd.Series(add[:, 1]).value_counts() if len(add) else pd.Series(dtype=int)
    conflicts = 0
    for a, b in add:
        if acc[a] >= 0 or b in taken_b or ca[a] > 1 or cb[b] > 1:
            conflicts += 1
            continue
        acc[a] = b
        taken_b.add(b)
    new = links.copy()
    new["status"] = np.where(acc[new["det_a"].to_numpy()] == new["det_b"].to_numpy(), "reference", "disagreement")
    m = dict(meta)
    m.pop("files", None)
    m["counts"] = dict(meta["counts"], reference_links=int((acc >= 0).sum()),
                       disagreements=int((new["status"] == "disagreement").sum()))
    m["audit"] = {"key_sha256": sha256_file(audit_dir / "audit_key.csv"), "annotator_A_sha256": sha256_file(pa),
                  "annotator_B_sha256": sha256_file(pb), "removed": int(len(both_diff)),
                  "added": int(len(add) - conflicts), "conflicts": int(conflicts),
                  "from_reference": json.loads((ref_dir / "reference.meta.json").read_text())["files"]}
    return consensus.write(consensus.Reference(acc, new, m), d, out_dir)
