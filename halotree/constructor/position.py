import os

from ..core import SmartBins
from ..dfield import HaloDensityField
from .constructor import Constructor

import numpy as np
import pandas as pd


class PositionConstructor(Constructor):
    def _PopulateHalos(self, m_halo, snap):
        bins = SmartBins(m_halo, self.n_min, self.d_min, 'log')
        return self.density_field.PopulateHalos(m_halo, bins, snap, self.rstate)


    def _MoveHalos(self, inds, m_self, m_desc, snap):
        x_h, y_h, z_h = self.density_field.SampleFromIndex(inds, rstate=self.rstate)
        return self.density_field.MoveHalos(x_h, y_h, z_h, inds, m_self, m_desc, snap)


    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        snap_max = self.snap_max
        snap_min = self.snap_min
        col_m_vir = self.col_m_vir
        col_x = self.col_x
        col_y = self.col_y
        col_z = self.col_z
        density_field = self.density_field
        n_grid = density_field.n_grid
        box_size = density_field.box_size
        z_fix = 0 if self.comoving else z
        rstate = self.rstate
        #
        # Sample positions for MC halos
        select_mc = df_req.index < 0
        grid_index = np.zeros(len(df_req), dtype='i4')
        m_mc = df_req.loc[select_mc, col_m_vir].values
        if snap == snap_max:
            if len(m_mc) > 0:
                x_h, y_h, z_h, grid_index[select_mc] = self._PopulateHalos(m_mc, snap)
        else:
            df_pool = pd.DataFrame.from_dict(
                self.pool, orient='index', columns=['GridIndex', 'MassDesc']
            )
            df_pool_mc = df_pool.loc[df_req.index[select_mc]]
            grid_index_mc = df_pool_mc['GridIndex'].values
            m_desc_mc = df_pool_mc['MassDesc'].values
            x_h, y_h, z_h, grid_index_mc = self._MoveHalos(grid_index_mc, m_mc, m_desc_mc, snap)
            grid_index[select_mc] = grid_index_mc
        #
        df_pos = pd.DataFrame(index=df_req.index[select_mc], columns=[col_x, col_y, col_z])
        if len(m_mc) > 0:
            df_pos[col_x] = x_h
            df_pos[col_y] = y_h
            df_pos[col_z] = z_h
            if z_fix != 0.:
                df_pos[[col_x, col_y, col_z]] /= (1. + z_fix)
        # Apply boundary condition to simulation halos
        df_inh[col_x] = density_field.BoundaryCondition(df_inh[col_x], z_fix)
        df_inh[col_y] = density_field.BoundaryCondition(df_inh[col_y], z_fix)
        df_inh[col_z] = density_field.BoundaryCondition(df_inh[col_z], z_fix)
        # Finalise output
        df_pos = df_inh.append(df_pos)
        #
        if snap > snap_min:
            # Compute grid indices for simulation halos
            select_sim = ~select_mc
            df_inh = df_inh.loc[df_req.index[select_sim]]
            grid_index[select_sim] = density_field.Grid2Index(
                df_inh[col_x], df_inh[col_y], df_inh[col_z], z_fix=z_fix
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
        comoving=True, rstate=None, kwargs_tqdm={}
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
            kwargs_tqdm=kwargs_tqdm
        )
        self.snap_max = snap_max
        self.snap_min = snap_min
        self.col_x = cols['x']
        self.col_y = cols['y']
        self.col_z = cols['z']
        self.col_m_vir = cols['m_vir']
        self.density_field = density_field
        self.n_min = n_min
        self.d_min = d_min
        self.comoving = comoving
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate
