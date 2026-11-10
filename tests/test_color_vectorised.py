"""C009: the vectorised CPU colour-distance matrices equal the original per-pair loop bit for bit."""
import numpy as np
import pytest

from cbtv import color


def _colours(rng, n, dtypes):
    out = []
    for k in range(n):
        dt = dtypes[k % len(dtypes)]
        out.append((rng.uniform(0, 255, 3).astype(dt), float(rng.uniform(0, 255))))
    return out


@pytest.mark.parametrize("t_dt,d_dt", [([np.float32], [np.float32]), ([np.float64], [np.float64]),
                                       ([np.float32, np.float64], [np.float64, np.float32])])
@pytest.mark.parametrize("saturation_weight,nonlinear_k", [(True, 2.0), (False, 0.0), (True, 0.0)])
def test_pair_matrices_bit_identical(t_dt, d_dt, saturation_weight, nonlinear_k):
    rng = np.random.default_rng(7)
    tracks, dets = _colours(rng, 37, t_dt), _colours(rng, 53, d_dt)
    a = color._pair_matrices_cpu(tracks, dets, saturation_weight, nonlinear_k)
    b = color._pair_matrices_scalar(tracks, dets, saturation_weight, nonlinear_k)
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype == np.float32
        assert np.array_equal(x, y)
