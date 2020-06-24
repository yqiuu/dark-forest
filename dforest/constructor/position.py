import os

from ..core import smart_bins
from .constructor import Constructor

import numpy as np
import pandas as pd
import tables


__all__ = ['PositionConstructor']


class PositionConstructor(Constructor):
    def __init__(self,
        cntr_ext, cntr_src, cols, density_field, n_min, d_min,
        snap_max, snap_min=0, postfix=None, comoving=True, kwargs_tqdm={}
    ):
        super().__init__(
            cntr_ext=cntr_ext,
            cntr_src=cntr_src,
            col_inh=[cols['x'], cols['y'], cols['z']],
            col_req=['ProgIndex', 'ProgNumber', cols['m_vir']],
            snap_max=snap_max,
            snap_min=snap_min,
            read_prog=['Index'],
            inherit=True,
            forward=False,
            postfix=postfix,
            message='Constructing coordinates',
            kwargs_tqdm=kwargs_tqdm
        )
        self.col_x = cols['x']
        self.col_y = cols['y']
        self.col_z = cols['z']
        self.col_m_vir = cols['m_vir']
        self.density_field = density_field
        self.n_min = n_min
        self.d_min = d_min
        self.comoving = comoving


    def _save_disps(self, inds, grid_inds, m_self, m_desc, disps, snap):
        hf = self.cntr_ext.hf
        group = hf.create_group(f'/Snapshot_{snap:03d}', 'Displacement')
        hf.create_carray(group, 'Index', obj=inds)
        hf.create_carray(group, 'GridIndex', obj=grid_inds)
        hf.create_carray(group, 'Mass', obj=m_self)
        hf.create_carray(group, 'MassDesc', obj=m_desc)
        hf.create_carray(group, 'DispX', obj=disps[0])
        hf.create_carray(group, 'DispY', obj=disps[1])
        hf.create_carray(group, 'DispZ', obj=disps[2])


    def _populate_halos(self, m_halo, snap):
        bins = smart_bins(m_halo, self.n_min, self.d_min, 'log')
        return self.density_field.populate_halos(m_halo, bins, snap)


    def _move_halos(self, inds, x_h, y_h, z_h, m_self, m_desc, snap):
        grid_inds = self.density_field.gt_vel.pos_to_index(x_h, y_h, z_h)
        pos_new, disps = self.density_field.move_halos(
            x_h, y_h, z_h, grid_inds, m_self, m_desc, snap, return_disps=True
        )
        self._save_disps(inds, grid_inds, m_self, m_desc, disps, snap)
        return pos_new


    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        snap_max = self.snap_max
        snap_min = self.snap_min
        col_m_vir = self.col_m_vir
        col_x = self.col_x
        col_y = self.col_y
        col_z = self.col_z
        density_field = self.density_field
        z_fix = 0 if self.comoving else z
        #
        # Sample positions for MC halos
        select_mc = df_req.index < 0
        n_halo = len(df_req)
        x_h = np.zeros(n_halo)
        y_h = np.zeros(n_halo)
        z_h = np.zeros(n_halo)
        m_mc = df_req.loc[select_mc, col_m_vir].values
        if snap == snap_max:
            if len(m_mc) > 0:
                x_mc, y_mc, z_mc, _ = self._populate_halos(m_mc, snap)
                x_h[select_mc] = x_mc
                y_h[select_mc] = y_mc
                z_h[select_mc] = z_mc
        else:
            df_pool = pd.DataFrame.from_dict(
                self.pool, orient='index', columns=['GridIndex', 'MassDesc']
            )
            df_pool_mc = df_pool.loc[df_req.index[select_mc]]
            x_desc, y_desc, z_desc \
                = self.density_field.gt_box.sample_from_inds(df_pool_mc['GridIndex'].values)
            m_desc_mc = df_pool_mc['MassDesc'].values
            inds = df_pool_mc.index.values
            x_mc, y_mc, z_mc, _ = self._move_halos(
                inds, x_desc, y_desc, z_desc, m_mc, m_desc_mc, snap
            )
            x_h[select_mc] = x_mc
            y_h[select_mc] = y_mc
            z_h[select_mc] = z_mc
        #
        df_pos = pd.DataFrame(index=df_req.index[select_mc], columns=[col_x, col_y, col_z])
        if len(m_mc) > 0:
            df_pos[col_x] = x_mc
            df_pos[col_y] = y_mc
            df_pos[col_z] = z_mc
            if z_fix != 0.:
                df_pos[[col_x, col_y, col_z]] /= (1. + z_fix)
        # Apply boundary condition to simulation halos
        df_inh[col_x] = density_field.check_bounds(df_inh[col_x], z_fix)
        df_inh[col_y] = density_field.check_bounds(df_inh[col_y], z_fix)
        df_inh[col_z] = density_field.check_bounds(df_inh[col_z], z_fix)
        #
        if snap > snap_min:
            # Compute grid indices for simulation halos
            select_sim = ~select_mc
            df_inh = df_inh.loc[df_req.index[select_sim]]
            x_h[select_sim] = df_inh[col_x]*(1. + z_fix)
            y_h[select_sim] = df_inh[col_y]*(1. + z_fix)
            z_h[select_sim] = df_inh[col_z]*(1. + z_fix)
            # Put grid indices into the pool
            pool = {}
            cond = df_req['ProgNumber'] > 0
            grid_inds = self.density_field.gt_box.pos_to_index(x_h[cond], y_h[cond], z_h[cond])
            for (idx, num, m_desc), g_idx, in zip(
                df_req[cond].itertuples(index=False), grid_inds
            ):
                for p_id in list_prog[idx:idx + num]:
                    if p_id < 0:
                        pool[p_id] = (g_idx, m_desc)
            self.pool = pool
        return df_pos
