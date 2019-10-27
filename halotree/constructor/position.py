import os

from ..core import SmartBins
from ..dfield import Index2Grid, Grid2Index, SampleFromIndex
from .constructor import Constructor

import numpy as np
import pandas as pd


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


    def _MovePosition(self, mass, grid_index, snap):
        # Load local variables
        snap_min = self.snap_min
        density_grid = self.density_grid
        n_grid = density_grid.n_grid
        box_size = density_grid.box_size
        ages = self.ages
        rstate = self.rstate
        #
        x_h, y_h, z_h = SampleFromIndex(grid_index, n_grid, box_size, rstate)
        if snap == snap_min:
            return x_h, y_h, z_h
        else:
            dt = ages[snap] - ages[snap - 1]
            bin_edges = SmartBins(mass, 100000, .1, 'log')
            bin_centres = np.sqrt(bin_edges[:-1]*bin_edges[1:])
            for m_c, m_l, m_u in zip(bin_centres, bin_edges[:-1], bin_edges[1:]):
                cond = (mass >= m_l) & (mass < m_u)
                x_h[cond], y_h[cond], z_h[cond], grid_index[cond] = density_grid.MovePosition(
                    m_c, x_h[cond], y_h[cond], z_h[cond], grid_index[cond], snap, dt
                )
        return x_h, y_h, z_h


    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        snap_max = self.snap_max
        snap_min = self.snap_min
        col_m_vir = self.col_m_vir
        col_x = self.col_x
        col_y = self.col_y
        col_z = self.col_z
        density_grid = self.density_grid
        n_grid = density_grid.n_grid
        box_size = density_grid.box_size
        comoving = self.comoving
        rstate = self.rstate
        #
        # Sample positions for MC halos
        select_mc = df_req.index < 0
        grid_index = np.zeros(len(df_req), dtype='i4')
        mass = df_req.loc[select_mc, col_m_vir].values
        if snap == snap_max:
            pos_x, pos_y, pos_z, grid_index[select_mc] = self._SamplePosition(mass, snap)
        else:
            df_pool = pd.DataFrame.from_dict(self.pool, orient='index', columns=['GridIndex'])
            grid_index_mc = df_pool.loc[df_req.index[select_mc], 'GridIndex'].values
            pos_x, pos_y, pos_z = self._MovePosition(mass, grid_index_mc, snap)
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
            if not comoving:
                box_size /= (1. + z)
            select_sim = ~select_mc
            df_inh = df_inh.loc[df_req.index[select_sim]]
            grid_index[select_sim] = Grid2Index(
                df_inh[col_x], df_inh[col_y], df_inh[col_z], n_grid, box_size
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
        self.ages = handler.ages
        self.n_bin = n_bin
        self.comoving = comoving
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate
