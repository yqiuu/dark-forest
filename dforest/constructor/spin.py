from ..logger import logger
from .constructor import Constructor

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


__all__ = [
    'KernelSpinSampler',
    'SpinConstructor'
]


def random_unit_sphere(n_sample, rstate):
    phi = rstate.uniform(0, 2*np.pi, n_sample)
    theta = np.arccos(rstate.uniform(-1, 1, n_sample))
    vector = np.zeros([3, n_sample])
    vector[0] = np.cos(phi)*np.sin(theta)
    vector[1] = np.sin(phi)*np.sin(theta)
    vector[2] = np.cos(theta)
    return vector


class KernelSpinSampler:
    def draw(self, n_halo, z):
        cntr = self.cntr
        snap_target = max(self.snap_fit, np.argmin(abs(self.redshifts - z)))
        while True:
            spin_sample = cntr[snap_target, 'lambda_B'][
                (cntr[snap_target, 'HostHaloID'] == cntr.host_id()) \
                & (cntr[snap_target, self.mass_key] > self.m_cut)
            ]
            if len(spin_sample) > self.n_min:
                break
            else:
                snap_target += 1
                self.snap_fit = snap_target
        cond = spin_sample != 0.
        if len(spin_sample) != np.count_nonzero(cond):
            logger.warning("Find halos with zero spin.")
        logspin = gaussian_kde(np.log10(spin_sample[cond])).resample(n_halo, self.rstate)
        return np.ravel(10**logspin)


    def __init__(self, cntr, mass_key, m_cut=0., n_min=100000, rstate=None):
        self.cntr = cntr
        self.redshifts = cntr.redshifts
        self.mass_key = mass_key
        self.m_cut = m_cut/cntr.mass_conversion
        self.n_min = n_min
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate
        self.snap_fit = 0 # Assume that the snapshot starts at zero


class SpinConstructor(Constructor):
    def __init__(self,
        cntr_ext, cntr_src, cols, spin_sampler, snap_max, snap_min=0,
        postfix=None, primary_spin=False, rstate=None, kwargs_tqdm={}
    ):
        super().__init__(
            cntr_ext=cntr_ext,
            cntr_src=cntr_src,
            col_inh=[cols['spin'], cols['L_x'], cols['L_y'], cols['L_z']],
            col_req=['ProgIndex', 'ProgNumber', cols['m_vir'], cols['r_vir'], cols['v_vir']],
            snap_min=snap_min,
            snap_max=snap_max,
            read_prog=['Index', cols['m_vir']],
            inherit=True,
            forward=True,
            postfix=postfix,
            message='Adding spin parameters',
            kwargs_tqdm=kwargs_tqdm
        )
        self.pool = {}
        self.snap_min = snap_min
        self.col_spin = cols['spin']
        self.col_m_vir = cols['m_vir']
        self.col_r_vir = cols['r_vir']
        self.col_v_vir = cols['v_vir']
        self.spin_sampler = spin_sampler
        self.primary_spin = primary_spin
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate
        self.mass_conversion = cntr_src.mass_conversion


    def _inherit_spin(self, prog):
        pool = self.pool
        m_max = -1.
        spin = np.nan
        for p_id, m in prog:
            if m > m_max:
                m_max = m
                spin = pool[p_id]
            del pool[p_id]
        return spin


    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        pool = self.pool
        snap_min = self.snap_min
        col_spin = self.col_spin
        col_m_vir = self.col_m_vir
        col_r_vir = self.col_r_vir
        col_v_vir = self.col_v_vir
        spin_sampler = self.spin_sampler
        primary_spin = self.primary_spin
        rstate = self.rstate
        mass_conversion = self.mass_conversion
        #
        df_req_mc = df_req[df_req.index < 0]
        spin = np.zeros(len(df_req_mc))
        if snap > snap_min and primary_spin:
            # Inherit spin parameters
            prog_pointer = df_req_mc[['ProgIndex', 'ProgNumber']].values
            cond = prog_pointer[:, -1] > 0
            spin[cond] = [
                self._inherit_spin(list_prog[idx:idx + num]) for idx, num in prog_pointer[cond]
            ]
            cond = ~cond
        else:
            cond = np.full(len(spin), True)
        # Draw random spin parameters
        spin[cond] = spin_sampler.draw(sum(cond), z)
        # Compute angular momenta
        L = np.sqrt(2.)/mass_conversion*np.asarray(
            df_req_mc[col_m_vir]*df_req_mc[col_r_vir]*df_req_mc[col_v_vir]*spin
        )
        L = random_unit_sphere(len(df_req_mc), rstate)*L
        #
        df_spin = pd.DataFrame(
            np.vstack([spin, L]).T, index = df_req_mc.index, columns = df_inh.columns
        )
        # Walk forwad
        if primary_spin:
            pool.update(
                {h_id:s for h_id, s in zip(df_req.index, df_spin.loc[df_req.index, col_spin])}
            )
        return df_spin
