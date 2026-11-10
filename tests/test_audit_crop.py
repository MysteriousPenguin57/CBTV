"""T035: audit.crop_pair must give the same output size for every item and put each image pixel at the right offset,
including when the crop window runs past the image edges (negative start index near (0,0) used to wrap / empty-slice,
and a centre past the right/bottom edge used to give a smaller crop)."""
import numpy as np
import pytest

from tools.realdata import audit

H, W, HALF, SCALE, MARGIN = 60, 80, 6, 4, 10


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(1, 255, (H, W, 3), dtype=np.uint8)   # no zeros, so padding (0) is distinguishable


def _expected_window(img, cx, cy):
    """(2*HALF+1)^2 window centred at (cx, cy), zero outside the image."""
    out = np.zeros((2 * HALF + 1, 2 * HALF + 1, 3), np.uint8)
    for j in range(2 * HALF + 1):
        for i in range(2 * HALF + 1):
            y, x = cy - HALF + j, cx - HALF + i
            if 0 <= y < H and 0 <= x < W:
                out[j, i] = img[y, x]
    return out


def _left_window(out):
    """Undo the panel layout: the left panel's image area, sampled once per source pixel."""
    s = (2 * HALF + 1) * SCALE
    area = out[MARGIN:MARGIN + s, MARGIN:MARGIN + s]
    return area[::SCALE, ::SCALE]


@pytest.mark.parametrize("cx,cy", [(40, 30), (0, 0), (2, 1), (-3, -2), (-1, 30), (40, -1), (W - 1, H - 1),
                                   (W - 2, 5), (W + 2, H + 3), (W, 30)])
def test_crop_pair_constant_size_and_pixel_offsets(cx, cy):
    f = _frame()
    ref = audit.crop_pair(f, f, 40.0, 30.0, 40.0, 30.0, HALF, scale=SCALE, margin=MARGIN)
    out = audit.crop_pair(f, f, float(cx), float(cy), float(cx), float(cy), HALF, scale=SCALE, margin=MARGIN)
    assert out.shape == ref.shape, "crop size changed at (%d, %d): %s vs %s" % (cx, cy, out.shape, ref.shape)
    np.testing.assert_array_equal(_left_window(out), _expected_window(f, cx, cy))
