from abc import ABC, abstractmethod

from .logger import logger

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf.growth_factor import GrowthFactor


__all__ = [
    'index_to_grid',
    'pos_to_index',
    'GridTool',
    'DensityField'
]


def index_to_grid(inds, n_grid):
    grid_z = inds%n_grid
    inds = inds//n_grid
    grid_y = inds%n_grid
    inds = inds//n_grid
    grid_x = inds%n_grid
    return grid_x, grid_y, grid_z


def pos_to_index(x, y, z, n_grid, box_size):
    bins = np.linspace(0, box_size, n_grid + 1)
    grid_x = np.digitize(x, bins) - 1
    grid_y = np.digitize(y, bins) - 1
    grid_z = np.digitize(z, bins) - 1
    return (grid_x*n_grid + grid_y)*n_grid + grid_z


class GridTool:
    def __init__(self, n_grid, box_size, rstate=None):
        self._n_grid = n_grid
        self._box_size = box_size
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate


    @property
    def n_grid(self):
        """
        Number of grids on a side.
        """
        return self._n_grid


    @property
    def box_size(self):
        """
        Box size. [Mpc]
        """
        return self._box_size


    def index_to_grid(self, inds):
        """
        Convert 1-D grid indices to 3-D grid indices.

        Parameters
        ----------
        inds : 1-D array
            Input 1-D grid indices.

        Returns
        -------
        x, y, z : 1-D array
            3-D grid indices.
        """
        return index_to_grid(inds, self.n_grid)


    def pos_to_index(self, xx, yy, zz, z_fix=0.):
        """
        Convert 3-D coordinates to 1-D grid indices.

        Parameters
        ----------
        xx, yy, zz : 1-D array
            Input 3-D coordinates.
        z_fix : float, optional
            Divide the box size by 1 + z_fix. Use this when coordinates are
            not comoving.

        Returns
        -------
        1-D array
            1-D grid indices.
        """
        return pos_to_index(xx, yy, zz, self.n_grid, self.box_size/(1. + z_fix))


    def sample_from_inds(self, inds):
        """
        Sample random points using 1-D grid indices. Each point is uniformly
        distributed in the corresponding cell.

        Parameters
        ----------
        inds : 1-D array
            Input 1-D grid indices.

        Returns
        -------
        x, y, z : 1-D array
            3-D coordinates.
        """
        n_grid = self.n_grid
        grid_size = self.box_size/n_grid
        rstate = self.rstate
        n_sample = len(inds)
        grid_x, grid_y, grid_z = self.index_to_grid(inds)
        x = (rstate.rand(n_sample) + grid_x)*grid_size
        y = (rstate.rand(n_sample) + grid_y)*grid_size
        z = (rstate.rand(n_sample) + grid_z)*grid_size
        return x, y, z


    def sample_from_field(self, den, n_sample, rstate=None):
        """
        Sample random points according to a given density field.

        Parameters
        ----------
        den : 1-D array
            Input normalised density field.
        n_sample : int
            Number of samples to generate.

        Returns
        -------
        x, y, z : 1-D array
            3-D coordinates.
        inds : 1-D array
            Corresponding 1-D grid indices.
        """
        inds = self.rstate.choice(len(den), n_sample, p=den)
        x, y, z = self.sample_from_inds(inds)
        return x, y, z, inds


class DensityField(ABC):
    def __init__(self, emf, redshifts, ages, n_grid, box_size, r_max=3., rstate=None):
        self.gt_box = GridTool(n_grid, box_size, rstate)
        self.emf = emf
        self.redshifts = redshifts
        self.ages = ages
        self.r_max = r_max
        # Compute growth factor
        gf = GrowthFactor(emf.cosmo)
        self.D = [gf.growth_factor(z) for z in redshifts]
        self.dDdt = spl(ages, self.D).derivative()(ages)


    @abstractmethod
    def read_density_field(self, snap):
        """
        Returns
        -------
        array
            Overdensity grid.
        """
        return np.array([0.])


    @abstractmethod
    def read_transform_field(self, snap):
        """
        Returns
        -------
        array [Mpc/Myr]
            Peculiar velocity grid.
        """
        return np.array([0.])


    def check_bounds(self, a, z_fix=0.):
        box_size = self.gt_box.box_size/(1. + z_fix)
        res = np.copy(a)
        res[res >= box_size] -= box_size
        res[res < 0.] += box_size
        return res


    def move(self, x, y, z, dx, dy, dz):
        x = self.check_bounds(x + dx)
        y = self.check_bounds(y + dy)
        z = self.check_bounds(z + dz)
        inds = self.gt_box.pos_to_index(x, y, z)
        return x, y, z, inds


    def populate_halos(self, m_halo, bins, snap):
        emf = self.emf
        z = self.redshifts[snap]
        v_cell = (self.gt_box.box_size/self.gt_box.n_grid*emf.cosmo.h)**3
        delta_dm = self.read_density_field(snap)

        n_halo = len(m_halo)
        x_h = np.zeros(n_halo, dtype='f8')
        y_h = np.zeros(n_halo, dtype='f8')
        z_h = np.zeros(n_halo, dtype='f8')
        inds = np.zeros(n_halo, dtype='i4')
        for l, u in zip(bins[:-1], bins[1:]):
            cond = (m_halo >= l) & (m_halo < u)
            sample_sub = m_halo[cond]
            m_median = np.median(sample_sub)
            den_halo = emf.halo_density(delta_dm, v_cell, m_median, z)
            den_halo /= np.sum(den_halo)
            x_h[cond], y_h[cond], z_h[cond], inds[cond] = \
                self.gt_box.sample_from_field(den_halo, len(sample_sub))
        return x_h, y_h, z_h, inds


    def move_halos(self, x_h, y_h, z_h, inds, m1, m2, snap1):
        """
        Parameters
        ----------
        m1: array [h^-1 M_solar]
            Mass of the halos to be moved.
        m2: array [h^-1 M_solar]
            Mass of the descendant halos.
        """
        logger.info("Moving halos at snapshot %d"%snap1)
        snap2 = snap1 + 1
        z1 = self.redshifts[snap1]
        z2 = self.redshifts[snap2]

        emf = self.emf
        emf.update(z=z1)
        b1 = spl(emf.m, emf.bias)(m1)
        emf.update(z=z2)
        b2 = spl(emf.m, emf.bias)(m2)

        vx2, vy2, vz2 = self.read_transform_field(snap2)
        vx1, vy1, vz1 = self.read_transform_field(snap1)

        dx = b1*vx1[inds] - b2*vx2[inds]
        dy = b1*vy1[inds] - b2*vy2[inds]
        dz = b1*vz1[inds] - b2*vz2[inds]

        dr = np.sqrt(dx*dx + dy*dy + dz*dz)
        cond = dr > self.r_max
        norm = self.r_max/dr[cond]
        dx[cond] *= norm
        dy[cond] *= norm
        dz[cond] *= norm

        dr_16, dr_50, dr_84 = np.percentile(dr, [16, 50, 84])
        logger.info(f"dr = {dr_50:.2f}/+{dr_84-dr_50:.2f}/-{dr_50-dr_16:.2f}")

        return self.move(x_h, y_h, z_h, dx, dy, dz)
