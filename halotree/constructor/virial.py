from .constructor import Constructor

import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM


class VirialConstructor(Constructor):
    def DeltaVir(self, z):
        x = self.cosmo.Om(z) - 1
        return 18.*np.pi*np.pi + 82.*x - 39.*x*x

    
    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        col_m_vir = self.col_m_vir
        col_r_vir = self.col_r_vir
        col_v_vir = self.col_v_vir
        delta_collapse = self.delta_collapse
        mode = self.mode
        cosmo = self.cosmo
        mass_conversion = self.mass_conversion
        density_unit = self.density_unit
        grav_const = self.grav_const
        # Choose halo density
        rho_crit = cosmo.critical_density(z).to(density_unit).value
        if mode == 'vir':
            rho_halo = self.DeltaVir(z)*rho_crit
        elif mode == 'crit':
            rho_halo = delta_collapse*rho_crit
        elif mode == 'mean':
            rho_halo = delta_collapse*cosmo.Om(z)*rho_crit
        else:
            raise ValueError("Choose mode from `vir`, `crit` and `mean`")
        # Compute virial radius
        func_radius = np.vectorize(
            lambda m_vir:(0.75/np.pi*m_vir/rho_halo)**(1./3.), 'd'
        )
        func_velocity = np.vectorize(
            lambda m_vir, r_vir:np.sqrt(grav_const*m_vir/r_vir), 'd'
        )
        df_req = df_req[df_req.index < 0]/mass_conversion
        df_vir = pd.DataFrame(index = df_req.index)
        if len(df_req) > 0:
            df_vir[col_r_vir] = func_radius(df_req)
            df_vir[col_v_vir] = func_velocity(df_req, df_vir)
        df_inh[col_v_vir] = func_velocity(*df_inh.values.T)
        df_inh.drop(columns = self.col_m_vir, inplace = True)
        df_vir = df_inh.append(df_vir)
        return df_vir


    def __init__(self,
        cntr_target, cntr_source, cols, snap_max, snap_min=0,
        delta_collapse=200., mode='crit', kwargs_tqdm={}
    ):
        super().__init__(
            cntr_target=cntr_target,
            cntr_source=cntr_source,
            col_inh=[cols['m_vir'], cols['r_vir']],
            col_req=[cols['m_hc']],
            snap_max=snap_max,
            snap_min=snap_min,
            read_prog=[],
            inherit=True,
            forward=True,
            desc_tqdm='Adding virial properties',
            kwargs_tqdm=kwargs_tqdm
        )
        self.col_m_vir = cols['m_vir']
        self.col_r_vir = cols['r_vir']
        self.col_v_vir = cols['v_vir']
        self.delta_collapse = delta_collapse
        self.mode = mode
        self.cosmo = FlatLambdaCDM(**cntr_source.cosmo_params)
        self.mass_conversion = cntr_source.mass_conversion
        self.density_unit = cntr_source.mass_unit/cntr_source.radius_unit**3
        self.grav_const = const.G.to(
            cntr_source.radius_unit/cntr_source.mass_unit*cntr_source.velocity_unit**2
        ).value
