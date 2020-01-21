from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf.growth_factor import GrowthFactor


def Index2Grid(inds, n_grid):
    grid_z = inds%n_grid
    inds = inds//n_grid
    grid_y = inds%n_grid
    inds = inds//n_grid
    grid_x = inds%n_grid
    return grid_x, grid_y, grid_z


class GridTool:
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


    def Index2Grid(self, inds):
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
        return Index2Grid(inds, self.n_grid)


    def Pos2Index(self, xx, yy, zz, z_fix=0.):
        """
        Convert 3-D coordinates to 1-D grid indices.

        Parameters
        ----------
        x, y, z : 1-D array
            Input 3-D coordinates.
        z_fix : float, optional
            Divide the box size by 1 + z_fix. Use this when coordinates are
            not comoving.

        Returns
        -------
        1-D array
            1-D grid indices.
        """
        n_grid = self.n_grid
        bins = np.linspace(0, self.box_size/(1. + z_fix), n_grid + 1)
        xx = np.digitize(xx, bins) - 1
        yy = np.digitize(yy, bins) - 1
        zz = np.digitize(zz, bins) - 1
        return (xx*n_grid + yy)*n_grid + zz


    def SampleFromIndex(self, inds, rstate=None):
        """
        Sample random points using 1-D grid indices. Each point is uniformly
        distributed in the corresponding cell.

        Parameters
        ----------
        inds : 1-D array
            Input 1-D grid indices.
        rstate : RandomState instance or None. default to None
            If RandomState instance, `rstate` is the random number generator.
            If None, the random number generator is the RandomState instance
            used by `numpy.random`.

        Returns
        -------
        x, y, z : 1-D array
            3-D coordinates.
        """
        if rstate is None:
            rstate = np.random.RandomState()
        #
        n_grid = self.n_grid
        grid_size = self.box_size/n_grid
        n_sample = len(inds)
        grid_x, grid_y, grid_z = self.Index2Grid(inds)
        x = (rstate.rand(n_sample) + grid_x)*grid_size
        y = (rstate.rand(n_sample) + grid_y)*grid_size
        z = (rstate.rand(n_sample) + grid_z)*grid_size
        return x, y, z


    def SampleFromField(self, den, n_sample, rstate=None):
        """
        Sample random points according to a given density field.

        Parameters
        ----------
        den : 1-D array
            Input normalised density field.
        n_sample : int
            Number of samples to generate.
        rstate : RandomState instance or None. default to None
            If RandomState instance, `rstate` is the random number generator.
            If None, the random number generator is the RandomState instance
            used by `numpy.random`.

        Returns
        -------
        x, y, z : 1-D array
            3-D coordinates.
        inds : 1-D array
            Corresponding 1-D grid indices.
        """
        if rstate is None:
            rstate = np.random.RandomState()
        #
        inds = rstate.choice(len(den), n_sample, p=den)
        x, y, z = self.SampleFromIndex(inds, rstate=rstate)
        return x, y, z, inds


    def __init__(self, n_grid, box_size):
        self._n_grid = n_grid
        self._box_size = box_size


class DensityField(ABC):
    @abstractmethod
    def ReadDensityField(self, snap):
        """
        Returns
        -------
        array
            Overdensity grid.
        """
        return np.array([0.])


    @abstractmethod
    def ReadVelocityField(self, snap):
        """
        Returns
        -------
        array [Mpc/Myr]
            Peculiar velocity grid.
        """
        return np.array([0.])


    def BoundaryCondition(self, a, z_fix=0.):
        box_size = self.gt_box.box_size/(1. + z_fix)
        res = np.copy(a)
        res[res >= box_size] -= box_size
        res[res < 0.] += box_size
        return res


    def Move(self, x, y, z, dx, dy, dz):
        x = self.BoundaryCondition(x + dx)
        y = self.BoundaryCondition(y + dy)
        z = self.BoundaryCondition(z + dz)
        inds = self.gt_box.Pos2Index(x, y, z)
        return x, y, z, inds


    def PopulateHalos(self, m_halo, bins, snap, rstate):
        emf = self.emf
        z = self.redshifts[snap]
        v_cell = (self.gt_box.box_size/self.gt_box.n_grid*emf.cosmo.h)**3
        delta_dm = self.ReadDensityField(snap)

        n_halo = len(m_halo)
        x_h = np.zeros(n_halo, dtype='f8')
        y_h = np.zeros(n_halo, dtype='f8')
        z_h = np.zeros(n_halo, dtype='f8')
        inds = np.zeros(n_halo, dtype='i4')
        for l, u in zip(bins[:-1], bins[1:]):
            cond = (m_halo >= l) & (m_halo < u)
            sample_sub = m_halo[cond]
            m_median = np.median(sample_sub)
            den_halo = emf.HaloDensity(delta_dm, v_cell, m_median, z)
            den_halo /= np.sum(den_halo)
            x_h[cond], y_h[cond], z_h[cond], inds[cond] = \
                self.gt_box.SampleFromField(den_halo, len(sample_sub), rstate)
        return x_h, y_h, z_h, inds


    def MoveHalos(self, x_h, y_h, z_h, inds, m1, m2, snap1):
        """
        Parameters
        ----------
        m1: array [h^-1 M_solar]
            Mass of the halos to be moved.
        m2: array [h^-1 M_solar]
            Mass of the descendant halos.
        """
        snap2 = snap1 + 1
        z1 = self.redshifts[snap1]
        z2 = self.redshifts[snap2]
        t1 = self.ages[snap1]
        t2 = self.ages[snap2]
        D1 = self.D[snap1]
        D2 = self.D[snap2]
        dDdt = self.dDdt[snap2]

        emf = self.emf
        emf.update(z=z1)
        b1 = spl(emf.m, emf.bias)(m1)
        emf.update(z=z2)
        b2 = spl(emf.m, emf.bias)(m2)

        factor = (b1*D1 - b2*D2)/dDdt*(1 + z2)
        vx, vy, vz = self.ReadVelocityField(snap2)
        return self.Move(x_h, y_h, z_h, vx[inds]*factor, vy[inds]*factor, vz[inds]*factor)


    def __init__(self, emf, redshifts, ages, n_grid, box_size):
        self.gt_box = GridTool(n_grid, box_size)
        self.emf = emf
        self.redshifts = redshifts
        self.ages = ages
        # Compute growth factor
        gf = GrowthFactor(emf.cosmo)
        self.D = [gf.growth_factor(z) for z in redshifts]
        self.dDdt = spl(ages, self.D).derivative()(ages)
