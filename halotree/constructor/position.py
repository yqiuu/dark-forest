import os
from abc import ABC, abstractmethod

from .constructor import Constructor

import numpy as np
import pandas as pd
from numpy import vectorize


def Index2Grid(index, n_grid):
    z = index%n_grid
    index //= n_grid
    y = index%n_grid
    index //= n_grid
    x = index%n_grid
    return x, y, z
Index2Grid = np.frompyfunc(Index2Grid, 2, 3)


def Grid2Index(x, y, z, n_grid):
    return (x*n_grid + y)*n_grid + z


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


    def SampleFromIndex(self, index, rstate):
        #
        n_dim = 3
        n_grid = self.n_grid
        box_size = self.box_size
        #
        n_sample = len(index)
        grid_size = box_size/n_grid
        grid_x, grid_y, grid_z = Index2Grid(index, n_grid)
        x = (rstate.rand(n_sample) + grid_x.astype('i4'))*grid_size
        y = (rstate.rand(n_sample) + grid_y.astype('i4'))*grid_size
        z = (rstate.rand(n_sample) + grid_z.astype('i4'))*grid_size
        return x, y, z


    def SamplePosition(self, m_halo, snap, n_sample, rstate):
        den_halo = self.HaloDensityGrid(m_halo, snap)
        index = rstate.choice(len(den_halo), n_sample, p=den_halo)
        x, y, z = self.SampleFromIndex(index, rstate)
        return x, y, z, index


    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass


class PositionConstructor(Constructor):
    def _SamplePosition(self, mass, snap):
        # Load local variables
        n_bin = self.n_bin
        density_grid = self.density_grid
        rstate = self.rstate
        #
        bin_edge = 10**np.linspace(np.log10(min(mass)), np.log10(max(mass)), n_bin)
        bin_centre = np.sqrt(bin_edge[:-1]*bin_edge[1:])
        n_sample = len(mass)
        pos_x = np.zeros(n_sample)
        pos_y = np.zeros(n_sample)
        pos_z = np.zeros(n_sample)
        grid_index = np.zeros(n_sample, dtype='i4')
        for m_c, m_l, m_u in zip(bin_centre, bin_edge[:-1], bin_edge[1:]):
            cond = (mass >= m_l) & (mass < m_u)
            pos_x[cond], pos_y[cond], pos_z[cond], grid_index[cond] = \
                density_grid.SamplePosition(m_c, snap, np.sum(cond), rstate)
        return pos_x, pos_y, pos_z, grid_index


    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        snap_max = self.snap_max
        snap_min = self.snap_min
        col_m_vir = self.col_m_vir
        col_x = self.col_x
        col_y = self.col_y
        col_z = self.col_z
        density_grid = self.density_grid
        comoving = self.comoving
        rstate = self.rstate
        # Sample positions for MC halos
        select_mc = df_req.index < 0
        grid_index = np.zeros(len(df_req), dtype='i4')
        if snap == snap_max:
            pos_x, pos_y, pos_z, grid_index[select_mc] = \
                self._SamplePosition(df_req.loc[select_mc, col_m_vir].values, snap)
        else:
            df_pool = pd.DataFrame.from_dict(self.pool, orient='index', columns=['GridIndex'])
            grid_index_mc = df_pool.loc[df_req.index[select_mc], 'GridIndex'].values
            pos_x, pos_y, pos_z = density_grid.SampleFromIndex(grid_index_mc, rstate)
            grid_index[select_mc] = grid_index_mc
        #
        df_pos = pd.DataFrame(index=df_req.index[select_mc])
        df_pos[col_x] = pos_x
        df_pos[col_y] = pos_y
        df_pos[col_z] = pos_z
        if not comoving:
            df_pos[[col_x, col_y, col_z]] /= (1. + z)
        df_pos = df_inh.append(df_pos)
        #
        if snap > snap_min:
            # Compute grid indices for simulation halos
            box_size = density_grid.box_size
            n_grid = density_grid.n_grid
            if comoving:
                bins = np.linspace(0, box_size, n_grid)
            else:
                bins = np.linspace(0., box_size/(1. + z), n_grid)
            select_sim = ~select_mc
            df_inh = df_inh.loc[df_req.index[select_sim]]
            grid_index[select_sim] = Grid2Index(
                np.digitize(df_inh[col_x], bins=bins),
                np.digitize(df_inh[col_y], bins=bins),
                np.digitize(df_inh[col_z], bins=bins),
                n_grid
            )
            # Put grid indices into the pool
            pool = {}
            cond = df_req['ProgNumber'] > 0
            for (idx, num), g_idx in zip(
                df_req.loc[cond, ['ProgIndex', 'ProgNumber']].values, grid_index[cond]
            ):
                for p_id in list_prog[idx:idx + num]:
                    if p_id < 0:
                        pool[p_id] = g_idx
            self.pool = pool
        return df_pos


    def __init__(self,
        handler, hc_path, hc_prefix, snap_min, snap_max, cols, density_grid, n_bin,
        comoving=True, rstate=None, progress_bar=True, kwargs_tqdm={}
    ):
        kwargs_tqdm.update(desc='Constructing coordinates')
        super().__init__(
            handler=handler,
            hc_path=hc_path,
            hc_prefix=hc_prefix,
            snap_min=snap_min,
            snap_max=snap_max,
            key_new='Pos',
            col_inh=[cols['x'], cols['y'], cols['z']],
            col_req=['ProgNumber', 'ProgIndex', cols['m_vir']],
            read_prog=['ID'],
            inherit=True,
            forward=False,
            progress_bar=progress_bar,
            kwargs_tqdm=kwargs_tqdm
        )
        self.snap_max = snap_max
        self.snap_min = snap_min
        self.col_x = cols['x']
        self.col_y = cols['y']
        self.col_z = cols['z']
        self.col_m_vir = cols['m_vir']
        self.density_grid = density_grid
        self.n_bin = n_bin
        self.comoving = comoving
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate
