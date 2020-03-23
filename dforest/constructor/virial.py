from .constructor import Constructor

import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM


__all__ = ['VirialConstructor']


class VirialConstructor(Constructor):
    def __init__(self,
        cntr_target, cntr_source, cols, snap_max, snap_min=0,
        postfix=None, mode='crit', delta_h=200., kwargs_tqdm={}
    ):
        super().__init__(
            cntr_target=cntr_target,
            cntr_source=cntr_source,
            col_inh=[cols['m_vir'], cols['r_vir']],
            col_req=[cols['m_main']],
            snap_max=snap_max,
            snap_min=snap_min,
            read_prog=[],
            inherit=False,
            forward=True,
            postfix=postfix,
            message='Adding virial properties',
            kwargs_tqdm=kwargs_tqdm
        )
        self.col_m_vir = cols['m_vir']
        self.col_r_vir = cols['r_vir']
        self.col_v_vir = cols['v_vir']
        self.mass_conversion = cntr_source.mass_conversion
        self.mode = mode
        self.delta_h = delta_h


    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        col_m_vir = self.col_m_vir
        col_r_vir = self.col_r_vir
        col_v_vir = self.col_v_vir
        rho_halo = self.cntr_source.rho_halo(z, self.mode, self.delta_h)
        G = self.cntr_source.G
        # Compute virial radius
        func_radius = np.vectorize(
            lambda m_vir:(0.75/np.pi*m_vir/rho_halo)**(1./3.), 'd'
        )
        func_velocity = np.vectorize(
            lambda m_vir, r_vir:np.sqrt(G*m_vir/r_vir), 'd'
        )
        df_req = df_req[df_req.index < 0]/self.mass_conversion
        df_vir = pd.DataFrame(index=df_req.index)
        if len(df_req) > 0:
            df_vir[col_r_vir] = func_radius(df_req)
            df_vir[col_v_vir] = func_velocity(df_req, df_vir)
        df_inh[col_v_vir] = func_velocity(*df_inh.values.T)
        df_inh.drop(columns=self.col_m_vir, inplace=True)
        df_vir = df_inh.append(df_vir)
        return df_vir
