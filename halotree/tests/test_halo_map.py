import pytest
import numpy as np
from halotree import FindMatch


@pytest.mark.parametrize("n_pts,n_grid", [(10000, 64)])
def test_HaloMap(n_pts, n_grid):
    rstate = np.random.RandomState(20070831)
    pos0 = rstate.rand(n_pts, 3)
    pos1 = rstate.rand(n_pts, 3)
    FindMatch(pos0, pos1, n_grid, 1., rstate)
    bins = np.linspace(0, 1, n_grid + 1)
    hist0 = np.histogramdd(pos0, bins = [bins]*3)[0]
    hist1 = np.histogramdd(pos1, bins = [bins]*3)[0]
    np.testing.assert_array_equal(hist0, hist1)
