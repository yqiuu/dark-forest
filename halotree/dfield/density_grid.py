from abc import ABC, abstractmethod

import numpy as np


def Index2Grid(index, n_grid):
    z = index%n_grid
    index //= n_grid
    y = index%n_grid
    index //= n_grid
    x = index%n_grid
    return x, y, z
Index2Grid = np.frompyfunc(Index2Grid, 2, 3)


def Grid2Index(x, y, z, n_grid, box_size):
    bins = np.linspace(0, box_size, n_grid + 1)
    x = np.digitize(x, bins) - 1
    y = np.digitize(y, bins) - 1 
    z = np.digitize(z, bins) - 1
    return (x*n_grid + y)*n_grid + z


def SampleFromIndex(index, n_grid, box_size, rstate=None):
    if rstate is None:
        rstate = np.random.RandomState()
    #
    n_dim = 3
    n_sample = len(index)
    grid_size = box_size/n_grid
    grid_x, grid_y, grid_z = Index2Grid(index, n_grid)
    x = (rstate.rand(n_sample) + grid_x.astype('i4'))*grid_size
    y = (rstate.rand(n_sample) + grid_y.astype('i4'))*grid_size
    z = (rstate.rand(n_sample) + grid_z.astype('i4'))*grid_size
    return x, y, z


class DensityGrid(ABC):
    @property
    @abstractmethod
    def n_grid(self):
        return 1


    @property
    @abstractmethod
    def box_size(self):
        return 0.


    @abstractmethod
    def HaloDensityGrid(self, m_halo, snap):
        return np.array([0.])


    def SamplePosition(self, m_halo, snap, n_sample, rstate):
        den_halo = self.HaloDensityGrid(m_halo, snap)
        index = rstate.choice(len(den_halo), n_sample, p=den_halo)
        x, y, z = SampleFromIndex(index, self.n_grid, self.box_size, rstate)
        return x, y, z, index


    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
