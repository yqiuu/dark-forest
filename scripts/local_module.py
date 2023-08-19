import sys

#sys.path.legend should point to where the dark-forest is located
# sys.path.append('/home/yqiu/repository/dark-forest/')
sys.path.append('/fred/oz013/balu/models/dark-forest')

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from dforest import Constructor

class SpecialConstructor(Constructor):
    def __init__(self,
        cntr_ext, cntr_src, snap_max, snap_min=0, postfix=None,
        mode='crit', delta_h=200., n_min=10000, rstate=None, kwargs_tqdm={}
    ):
        super().__init__(
            cntr_ext=cntr_ext,
            cntr_src=cntr_src,
            col_inh=['Mass_FOF', 'Mass_tot', 'Lx', 'Ly', 'Lz'],
            col_req=['Mass_FOF'],
            snap_max=snap_max,
            snap_min=snap_min,
            read_prog=[],
            inherit=False,
            forward=False,
            postfix=postfix,
            message='Adding virial properties',
            kwargs_tqdm=kwargs_tqdm
        )
        self.mass_conversion = cntr_src.mass_conversion
        self.mode = mode
        self.delta_h = delta_h
        self.n_min = n_min
        print(n_min)
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate


    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        #
        rho_halo = self.cntr_src.rho_halo(z, self.mode, self.delta_h)
        G = self.cntr_src.G
        def virial_radius(m):
            return (0.75/np.pi*m/rho_halo)**(1./3.)
        def virial_velocity(m, r):
            return np.sqrt(G*m/r)
        def spin_parameter(jx, jy, jz, m, r, v):
            j = np.sqrt(jx*jx + jy*jy + jz*jz)
            return j/(np.sqrt(2.)*m*r*v)
        # Compute virial properties for NS halos
        df_vir_ns = pd.DataFrame(index=df_inh.index)
        n_halo_ns = len(df_inh)
        if n_halo_ns > 0:
            m_vir = np.zeros(n_halo_ns)
            cond = df_inh['Mass_FOF'].values != 0
            m_vir[cond] = df_inh.loc[cond, 'Mass_FOF'].values
            cond = ~cond
            m_vir[cond] = df_inh.loc[cond, 'Mass_tot'].values
            r_vir = virial_radius(m_vir)
            v_vir = virial_velocity(m_vir, r_vir)
            Lx = df_inh['Lx'].values
            Ly = df_inh['Ly'].values
            Lz = df_inh['Lz'].values
            spin = spin_parameter(Lx, Ly, Lz, m_vir, r_vir, v_vir)
            df_vir_ns['R_200crit_FOF'] = r_vir
            df_vir_ns['V_200crit_FOF'] = v_vir
            df_vir_ns['Lx'] = Lx
            df_vir_ns['Ly'] = Ly
            df_vir_ns['Lz'] = Lz
            df_vir_ns['Spin_200crit_FOF'] = spin
        # Compute virial properties for MC halos
        df_req = df_req[df_req.index < 0]/self.mass_conversion
        df_vir_mc = pd.DataFrame(index=df_req.index)
        n_halo_mc = len(df_req)
        if n_halo_mc > 0:
            m_vir = df_req['Mass_FOF'].values
            r_vir = virial_radius(m_vir)
            v_vir = virial_velocity(m_vir, r_vir)
            df_vir_mc['R_200crit_FOF'] = r_vir
            df_vir_mc['V_200crit_FOF'] = v_vir
            if n_halo_ns > self.n_min:
                kernel = gaussian_kde(np.log10(spin))
                self.kernel = kernel
            else:
                kernel = self.kernel
            spin = 10**np.ravel(kernel.resample(n_halo_mc, self.rstate))
            df_vir_mc['Lx'] = 0.
            df_vir_mc['Ly'] = 0.
            df_vir_mc['Lz'] = np.sqrt(2)*m_vir*r_vir*v_vir*spin
            df_vir_mc['Spin_200crit_FOF'] = spin
        #
        df_vir = df_vir_ns.append(df_vir_mc)
        return df_vir
