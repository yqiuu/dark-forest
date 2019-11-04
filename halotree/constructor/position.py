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
        x_h = np.zeros(n_sample)
        y_h = np.zeros(n_sample)
        y_z = np.zeros(n_sample)
        grid_index = np.zeros(n_sample, dtype='i4')
        for m_c, m_l, m_u in zip(bin_centre, bin_edge[:-1], bin_edge[1:]):
            cond = (mass >= m_l) & (mass < m_u)
            x_h[cond], y_h[cond], y_z[cond], grid_index[cond] = \
                density_grid.SamplePosition(m_c, snap, np.sum(cond), rstate)
        return x_h, y_h, y_z, grid_index


    def _MovePosition(self, inds, m_self, m_desc, snap):
        # Load local variables
        density_grid = self.density_grid
        n_grid = density_grid.n_grid
        box_size = density_grid.box_size
        rstate = self.rstate
        #
        x_h, y_h, z_h = SampleFromIndex(inds, n_grid, box_size, rstate)
        return density_grid.MovePosition(x_h, y_h, z_h, inds, m_self, m_desc, snap - 1)


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
        m_mc = df_req.loc[select_mc, col_m_vir].values
        if snap == snap_max:
            x_h, y_h, y_z, grid_index[select_mc] = self._SamplePosition(m_mc, snap)
        else:
            df_pool = pd.DataFrame.from_dict(
                self.pool, orient='index', columns=['GridIndex', 'MassDesc']
            )
            df_pool_mc = df_pool.loc[df_req.index[select_mc]]
            grid_index_mc = df_pool_mc['GridIndex'].values
            m_desc_mc = df_pool_mc['MassDesc'].values
            x_h, y_h, y_z, grid_index_mc \
                = self._MovePosition(grid_index_mc, m_mc, m_desc_mc, snap)
            grid_index[select_mc] = grid_index_mc
        #
        df_pos = pd.DataFrame(index=df_req.index[select_mc])
        df_pos[col_x] = x_h
        df_pos[col_y] = y_h
        df_pos[col_z] = y_z
        if not comoving:
            df_pos[[col_x, col_y, col_z]] /= (1. + z)
        # Apply boundary condition to simulation halos
        if comoving:
            z_fix = 0.
        else:
            z_fix = z
        df_inh[col_x] = density_grid.BoundaryCondition(df_inh[col_x], z_fix)
        df_inh[col_y] = density_grid.BoundaryCondition(df_inh[col_y], z_fix)
        df_inh[col_z] = density_grid.BoundaryCondition(df_inh[col_z], z_fix)
        # Finalise output
        df_pos = df_inh.append(df_pos)
        #
        if snap > snap_min:
            if not comoving:
                box_size /= (1. + z)
            # Compute grid indices for simulation halos
            select_sim = ~select_mc
            df_inh = df_inh.loc[df_req.index[select_sim]]
            grid_index[select_sim] = Grid2Index(
                df_inh[col_x], df_inh[col_y], df_inh[col_z], n_grid, box_size
            )
            # Put grid indices into the pool
            pool = {}
            cond = df_req['ProgNumber'] > 0
            for (idx, num, m_desc), g_idx in zip(
                df_req[cond].itertuples(index=False), grid_index[cond]
            ):
                for p_id in list_prog[idx:idx + num]:
                    if p_id < 0:
                        pool[p_id] = (g_idx, m_desc)
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
