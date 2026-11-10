"""Amendment 2 audit design: allocation, inclusion weights, A/B overlap, seed determinism, HT report."""
import numpy as np
import pandas as pd
import pytest

from tools.realdata import audit


def _window(n_ref=900, n_dis=700, seed=3, n_sets=3):
    """A fabricated window: detections (for d and the frame range) + candidate links with proposers and a truth."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"frame": np.repeat(np.arange(100), 20), "x": rng.uniform(0, 500, 2000),
                      "y": rng.uniform(0, 500, 2000), "r": 5.0})
    d["det_id"] = np.arange(len(d))
    n = n_ref + n_dis
    fa = rng.integers(0, 99, n)
    xa, ya = rng.uniform(0, 500, n), rng.uniform(0, 500, n)
    step = rng.gamma(2.0, 3.0, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    props = np.array(["lap|nn|relax", "lap", "nn", "relax", "lap|nn", "nn|relax", "lap|relax"])
    links = pd.DataFrame({"frame_a": fa, "det_a": np.arange(n), "x_a": xa, "y_a": ya, "frame_b": fa + 1,
                          "det_b": np.arange(n) + n, "x_b": xa + step * np.cos(ang), "y_b": ya + step * np.sin(ang),
                          "status": ["reference"] * n_ref + ["disagreement"] * n_dis,
                          "proposers": ["lap|nn|relax"] * n_ref + list(rng.choice(props[1:1 + n_sets], n_dis,
                                                                                  p=_probs(n_sets)))})
    # truth: 2% of reference links wrong; disagreements true with a probability that depends on the stratum
    truth = np.ones(n, bool)
    truth[:n_ref] = rng.random(n_ref) > 0.02
    pt = np.where(step[n_ref:] < 4, 0.6, 0.15)
    truth[n_ref:] = rng.random(n_dis) < pt
    links["truth"] = truth
    return links, d


def _probs(k):
    p = np.array([.4, .25, .15, .1, .06, .04][:k])
    return p / p.sum()


def test_minimums_kept_when_they_exceed_n():
    """Many small strata: >= 10 each wins over the total (documented)."""
    links, d = _window(n_sets=6)
    key, st = audit.sample_a2(links, d)
    dis = st[st["design"] == "disagreement"]
    assert len(dis) > 30
    assert (dis["sampled"] >= np.minimum(10, dis["population"])).all()
    assert (key["design"] == "disagreement").sum() == dis["sampled"].sum() > 300


def test_allocate_min_proportional_min10_exact_total():
    sizes = pd.Series({"a": 1000, "b": 300, "c": 40, "d": 7, "e": 0})
    a = audit.allocate_min(sizes, 300, n_min=10)
    assert "e" not in a.index                                     # empty strata get nothing
    assert a["d"] == 7                                            # smaller than the minimum: all of it
    assert a["c"] >= 10 and (a <= sizes[a.index]).all()
    assert a.sum() == 300
    assert a["a"] > a["b"] > a["c"]                               # proportional beyond the minimum
    assert audit.allocate_min(sizes, 5000).to_dict() == {"a": 1000, "b": 300, "c": 40, "d": 7}


def test_weights_sum_to_stratum_sizes_and_population():
    links, d = _window()
    key, st = audit.sample_a2(links, d, n_random=100, n_dis=300, seed=audit.A2_SEED)
    assert (key["design"] == "random").sum() == 100
    assert (key["design"] == "disagreement").sum() == 300
    rnd = key[key["design"] == "random"]
    assert rnd["w"].sum() == pytest.approx((links["status"] == "reference").sum())
    pop = st.set_index("stratum")["population"]
    for h, g in key[key["design"] == "disagreement"].groupby("stratum"):
        assert g["w"].sum() == pytest.approx(pop[h])
        assert len(g) >= min(10, pop[h])
    assert st.loc[st["design"] == "disagreement", "population"].sum() == (links["status"] == "disagreement").sum()
    assert not key["members"].str.contains("cbtv", case=False).any()


def test_b_overlap_is_a_quarter_and_files(tmp_path):
    links, d = _window()
    key, st = audit.sample_a2(links, d)
    assert key["in_b"].sum() == round(0.25 * len(key))
    out = audit.export(key, st, tmp_path / "pkt", video=None)
    a = pd.read_csv(out / "annotator_A.csv", dtype={"item_id": str})
    b = pd.read_csv(out / "annotator_B.csv", dtype={"item_id": str})
    assert set(a["item_id"]) == set(key["item_id"])
    assert set(b["item_id"]) == set(key.loc[key["in_b"], "item_id"])
    assert list(b["item_id"]) != list(a.loc[a["item_id"].isin(b["item_id"]), "item_id"])     # independent order


def test_seed_determinism():
    links, d = _window()
    k1, s1 = audit.sample_a2(links, d, seed=audit.A2_SEED)
    k2, s2 = audit.sample_a2(links, d, seed=audit.A2_SEED)
    pd.testing.assert_frame_equal(k1, k2)
    pd.testing.assert_frame_equal(s1, s2)
    k3, _ = audit.sample_a2(links, d, seed=audit.A2_SEED + 1)
    assert set(k3["det_a"]) != set(k1["det_a"])


def _oracle(key, truth_by_det_a):
    lab = np.where(key["det_a"].map(truth_by_det_a), "same", "different")
    return pd.DataFrame({"item_id": key["item_id"], "judgement": lab})


def test_report_a2_oracle_recovers_truth_within_ci():
    links, d = _window()
    truth = links.set_index("det_a")["truth"]
    key, st = audit.sample_a2(links, d)
    ann = _oracle(key, truth)
    rep = audit.report_a2(key, st, ann, ann[ann["item_id"].isin(key.loc[key["in_b"], "item_id"])], n_boot=500)
    ref = links[links["status"] == "reference"]
    dis = links[links["status"] == "disagreement"]
    true_err = 1 - ref["truth"].mean()
    true_dis = int(dis["truth"].sum())
    right = len(ref) - (~ref["truth"]).sum()
    true_cov = right / (right + true_dis)
    lo, hi = rep["consensus_error_cp95"]
    assert lo <= true_err <= hi
    lo, hi = rep["dis_true_hat_ci95"]
    assert lo <= true_dis <= hi
    lo, hi = rep["coverage_ci95"]
    assert lo <= true_cov <= hi
    assert rep["kappa"] == 1.0 and rep["unresolved"] == 0


def test_adjudication_resolves_overlap_disagreements(tmp_path):
    links, d = _window()
    key, st = audit.sample_a2(links, d)
    a = pd.DataFrame({"item_id": key["item_id"], "judgement": "same"})
    b = a[a["item_id"].isin(key.loc[key["in_b"], "item_id"])].copy()
    flip = b["item_id"].iloc[:5]
    b.loc[b["item_id"].isin(flip), "judgement"] = "different"
    rep = audit.report_a2(key, st, a, b, n_boot=50)
    assert rep["overlap_disagreements"] == 5 and rep["unresolved"] == 5
    sheet = pd.read_csv(audit.adjudication_sheet(key, a, b, tmp_path / "adj.csv"), dtype={"item_id": str})
    assert set(sheet["item_id"]) == set(flip)
    sheet["adjudicated"] = "different"
    rep2 = audit.report_a2(key, st, a, b, sheet, n_boot=50)
    assert rep2["unresolved"] == 0


def test_design_note_reports_realised_n_and_reason():
    links, d = _window(n_sets=6)
    _, st = audit.sample_a2(links, d)
    txt = audit.design_note(st)
    assert "nominal 300" in txt and "minimum of 10 per non-empty stratum" in txt
    links, d = _window()
    _, st = audit.sample_a2(links, d)
    assert "exceed" not in audit.design_note(st)
