import numpy as np
from numpy.testing import assert_array_equal
from halotree import Index2Grid, Grid2Index, SampleFromIndex


def test_SampleFromIndex():
    rstate = np.random.RandomState(20070831)
    n_grid = 64
    box_size = 1.
    x0, y0, z0 = rstate.rand(3, 10**5)
    index = Grid2Index(x0, y0, z0, n_grid, box_size)
    x1, y1, z1 = SampleFromIndex(index, n_grid, box_size, rstate=rstate)
    bins = np.linspace(0, 1., n_grid + 1)
    hist0 = np.histogramdd((x0, y0, z0), bins=[bins]*3)[0]
    hist1 = np.histogramdd((x1, y1, z1), bins=[bins]*3)[0]
    assert_array_equal(hist0, hist1)
