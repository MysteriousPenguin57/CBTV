"""C003 unit tests (BASELINES_SPEC section 1.6). Fixed seeds; whole file well under 60 s."""
import csv
import os
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
from relaxation_ptv import RelaxParams, link_pair, track  # noqa: E402

R = 5.0          # detection radius -> d = 10 px
D = 2 * R


def points(n, seed, spacing=2.4 * D, jitter=0.2 * D):
    """n points on a jittered square grid: min separation >= spacing - 2*jitter (= 2d by default)."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(np.sqrt(n)))
    g = np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2)[:n] * spacing
    return g + rng.uniform(-jitter, jitter, g.shape) + 10 * D


def two_frames(A, B, colA=None, colB=None):
    return [(0, A, np.full(len(A), R), colA), (1, B, np.full(len(B), R), colB)]


def links(rows):
    """{index in frame 0: index in frame 1} for detections that kept their track id (rows keep input order)."""
    f0 = [r[1] for r in rows if r[0] == 0]
    f1 = [r[1] for r in rows if r[0] == 1]
    pos = {tid: i for i, tid in enumerate(f0)}
    return {pos[tid]: j for j, tid in enumerate(f1) if tid in pos}


def greedy_nn(A, B):
    """Globally greedy nearest-neighbour linking (ascending distance)."""
    dist = np.hypot(*(A[:, None, :] - B[None, :, :]).transpose(2, 0, 1))
    used_a, used_b, out = set(), set(), {}
    for flat in np.argsort(dist, axis=None):
        i, j = np.unravel_index(flat, dist.shape)
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); out[int(i)] = int(j)
    return out


def test_rigid_translation():
    A = points(50, seed=1)
    B = A + np.array([0.6 * D, -0.4 * D])
    rows, _ = track(two_frames(A, B))
    got = links(rows)
    assert sum(got.get(i) == i for i in range(50)) == 50
    # P* of every accepted link
    matches, _ = link_pair(A, B, D, 1.1 * D)
    assert len(matches) == 50 and min(m[2] for m in matches) > 0.9


def test_translation_plus_rotation():
    A = points(50, seed=2)
    th = np.deg2rad(6.0)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = A.mean(axis=0)
    B = (A - c) @ rot.T + c + np.array([0.6 * D, -0.4 * D])
    got = links(track(two_frames(A, B))[0])
    assert sum(got.get(i) == i for i in range(50)) >= 49


def test_swap_trap_cluster_context():
    # p, q 0.8d apart; the rigid 6-point cluster moves by ~0.85d along p->q, so p' lands next to q.
    A = np.array([[0, 0], [0.8, 0], [0.4, 2.0], [-1.5, 1.0], [2.0, -1.2], [0.3, -2.1]]) * D + 200
    B = A + np.array([0.85, 0.02]) * D
    assert greedy_nn(A, B)[1] != 1                       # greedy NN takes q -> p'
    got = links(track(two_frames(A, B))[0])
    assert got.get(0) == 0 and got.get(1) == 1           # relaxation: p -> p', q -> q'


@pytest.mark.parametrize("mode", ["none", "gate", "weight"])
def test_swap_trap_colour_no_context(mode):
    # Two particles, no cluster, truly crossing (opposite motion); different colour bins.
    A = np.array([[0.0, 0.0], [0.8, 0.0]]) * D + 200
    B = np.array([[0.85, 0.1], [-0.05, 0.1]]) * D + 200   # B[0] = p', B[1] = q'
    col = np.array([0, 1])
    rows, _ = track(two_frames(A, B, col, col), RelaxParams(colour_mode=mode))
    got = links(rows)
    correct = got.get(0) == 0 and got.get(1) == 1
    assert correct == (mode != "none")


def test_birth_death():
    A = points(53, seed=4)[:50]
    keep = np.arange(3, 50)                               # delete A[0..2] from B
    far = np.array([[60, 60], [62, 70], [70, 62]]) * D    # 3 far-away births
    B = np.vstack([A[keep] + np.array([0.5 * D, 0.3 * D]), far])
    rows, info = track(two_frames(A, B))
    got = links(rows)
    assert sum(got.get(int(a)) == b for b, a in enumerate(keep)) == 47
    assert len(got) == 47                                  # 3 deaths, 3 births (new ids)
    assert info["n_tracks"] == 53


def test_performance_534():
    A = points(534, seed=5)
    B = A + np.array([0.6 * D, 0.2 * D])
    t0 = time.process_time()
    got = links(track(two_frames(A, B))[0])
    assert time.process_time() - t0 < 10.0
    assert sum(got.get(i) == i for i in range(534)) == 534


def test_output_columns(tmp_path):
    A = points(20, seed=6)
    det = tmp_path / "dets.csv"
    with open(det, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "x", "y", "r", "score"])
        for t in range(3):
            for x, y in A + t * np.array([0.5 * D, 0]):
                w.writerow([t, x, y, R, 1.0])
    out = tmp_path / "tracks.csv"
    res = subprocess.run([sys.executable, os.path.join(PKG, "tracker_relax.py"), "--detections-csv", str(det),
                          "--output-csv", str(out), "--colour-mode", "none"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "[RELAX] linked in" in res.stdout
    rows = list(csv.reader(open(out)))
    assert rows[0] == ["frame", "track_id", "x", "y", "r", "missed"]
    body = rows[1:]
    assert len(body) == 60 and all(int(r[1]) >= 1 and r[5] == "0" for r in body)
    assert len({r[1] for r in body}) == 20
