from .constructor import Constructor

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


class VirialConstructor(Constructor):
    def DeltaVir(self, z):
        x = self.cosmo.Om(z) - 1
        return 18.*np.pi*np.pi + 82.*x - 39.*x*x

    
    def AddProperty(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        delta_collapse = self.delta_collapse
        mode = self.mode
        cosmo = self.cosmo
        mass_conversion = self.mass_conversion
        density_unit = self.density_unit
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
        #
        func_radius = np.vectorize(
            lambda m_vir:(0.75/np.pi*mass_conversion*m_vir/rho_halo)**(1./3.), df_inh.dtypes
        )
        df_req = df_req[df_req.index < 0]
        df_vir = pd.DataFrame(func_radius(df_req), index = df_req.index, columns = df_inh.columns)
        df_vir = df_inh.append(df_vir)
        return df_vir


    def __init__(self,
        handler, hc_path, hc_prefix, snap_min, snap_max, hdf_key, mass_key, radius_key,
        delta_collapse = 200., mode = 'crit', progress_bar = True, kwargs_tqdm = {}
    ):
        kwargs_tqdm.update(desc = 'Adding virial radii')
        super().__init__(
            handler = handler,
            hc_path = hc_path,
            hc_prefix = hc_prefix,
            snap_min = snap_min,
            snap_max = snap_max,
            key_new = hdf_key,
            col_new = [radius_key],
            col_req = [mass_key],
            inherit = True,
            read_prog = False,
            forward = True,
            progress_bar = progress_bar,
            kwargs_tqdm = kwargs_tqdm
        )
        self.delta_collapse = delta_collapse
        self.mode = mode
        self.cosmo = FlatLambdaCDM(**handler.cosmo_params)
        self.mass_conversion = handler.mass_conversion
        self.density_unit = handler.mass_unit/handler.radius_unit**3
