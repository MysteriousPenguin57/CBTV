"""--piv-ambiguity-disc: the half-width (px) of the square excluded around the PIV correlation peak when the second
peak is sought (cbtv.piv.peak_sharpness_to_sigma). CPU only, synthetic correlation maps, no tracker run."""
import numpy as np
import pytest

from cbtv.config import build_parser, validate_args
from cbtv.piv import peak_sharpness_to_sigma


def _sigma_fixed3(corr_2d, peak_y, peak_x, *, base_sigma=2.0, max_sigma=8.0):
    """The function as it was before the option existed (disc hard-coded to 3 px), kept verbatim as the reference."""
    peak = float(corr_2d[peak_y, peak_x])
    if not np.isfinite(peak) or peak < 0.2:
        return float(max_sigma)
    conf = min(max((peak - 0.3) / (0.85 - 0.3), 0.0), 1.0)
    h, w = corr_2d.shape
    cc = corr_2d.copy()
    y0, y1 = max(0, peak_y - 3), min(h, peak_y + 4)
    x0, x1 = max(0, peak_x - 3), min(w, peak_x + 4)
    cc[y0:y1, x0:x1] = -1.0
    second = float(cc.max())
    if second > 0.0:
        amb = second / max(peak, 1e-6)
        conf *= min(max((0.9 - amb) / (0.9 - 0.5), 0.0), 1.0)
    return float(max_sigma - (max_sigma - base_sigma) * conf)


def _gauss_peak(size=41, width=6.0, height=0.95, cy=20, cx=20):
    y, x = np.mgrid[0:size, 0:size]
    return (height * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * width ** 2))).astype(np.float32)


def test_default_equals_previous_behaviour():
    rng = np.random.default_rng(0)
    maps = [_gauss_peak(width=w) for w in (1.5, 3.0, 6.0)]
    maps += [np.clip(rng.normal(0.2, 0.3, (33, 33)), -1, 1).astype(np.float32) for _ in range(20)]
    maps.append(_gauss_peak(cy=1, cx=39))                       # peak at the border: clipped square
    for m in maps:
        py, px = np.unravel_index(int(np.argmax(m)), m.shape)
        assert peak_sharpness_to_sigma(m, py, px) == _sigma_fixed3(m, py, px)
        assert peak_sharpness_to_sigma(m, py, px, disc=3) == _sigma_fixed3(m, py, px)


def test_wider_disc_removes_the_peak_flank():
    # A broad single peak (as at twice the image scale): with the default 3 px square its own flank 4 px away is
    # taken for a second peak (ratio 0.80 -> confidence x0.25); a 12 px square excludes the flank.
    m = _gauss_peak(width=6.0)
    s_default = peak_sharpness_to_sigma(m, 20, 20)
    s_wide = peak_sharpness_to_sigma(m, 20, 20, disc=12)
    assert s_default == pytest.approx(6.5, abs=0.05)
    assert s_wide == pytest.approx(2.0, abs=1e-6)
    # a narrow peak is unaffected by the choice
    n = _gauss_peak(width=1.0)
    assert peak_sharpness_to_sigma(n, 20, 20) == pytest.approx(peak_sharpness_to_sigma(n, 20, 20, disc=12))


def test_cli_option_default_and_validation():
    ap = build_parser()
    assert ap.get_default("piv_ambiguity_disc") == 3
    args = ap.parse_args(["--input", "v.mp4", "--output-video", "o.mp4", "--output-csv", "o.csv",
                          "--piv-ambiguity-disc", "6"])
    assert args.piv_ambiguity_disc == 6
    args.piv_ambiguity_disc = 0
    with pytest.raises(SystemExit, match="piv-ambiguity-disc"):
        validate_args(args)
