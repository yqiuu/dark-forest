from .constructor import Constructor

import numpy as np
import pandas as pd


def RandomUnitSphere(n_sample, rstate):
    phi = rstate.uniform(0, 2*np.pi, n_sample)
    theta = np.arccos(rstate.uniform(-1, 1, n_sample))
    vector = np.zeros([3, n_sample])
    vector[0] = np.cos(phi)*np.sin(theta)
    vector[1] = np.sin(phi)*np.sin(theta)
    vector[2] = np.cos(theta)
    return vector


class SpinConstructor(Constructor):
    def _InheritSpin(self, prog):
        pool = self.pool
        m_max = -1.
        spin = np.nan
        for p_id, m in prog:
            if m > m_max:
                m_max = m
                spin = pool[p_id]
            del pool[p_id]
        return spin


    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
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
        if snap > snap_min:
            # Inherit spin parameters
            prog_pointer = df_req_mc[['ProgIndex', 'ProgNumber']].values
            if primary_spin:
                cond = prog_pointer[:, -1] > 0
            else:
                cond = prog_pointer[:, -1] == 1
            spin[cond] = [
                self._InheritSpin(list_prog[idx:idx + num]) for idx, num in prog_pointer[cond]
            ]
            cond = ~cond
        else:
            cond = np.full(len(spin), True)
        # Draw random spin parameters
        spin[cond] = spin_sampler.Draw(sum(cond), z)
        # Compute angular momenta
        L = np.sqrt(2.)/mass_conversion*np.asarray(
            df_req_mc[col_m_vir]*df_req_mc[col_r_vir]*df_req_mc[col_v_vir]*spin
        )
        L = RandomUnitSphere(len(df_req_mc), rstate)*L
        #
        df_spin = pd.DataFrame(
            np.vstack([spin, L]).T, index = df_req_mc.index, columns = df_inh.columns
        )
        df_spin = df_inh.append(df_spin)
        # Walk forwad
        pool.update(
            {h_id:s for h_id, s in zip(df_req.index, df_spin.loc[df_req.index, col_spin])}
        )
        return df_spin


    def __init__(self,
        handler, hc_path, hc_prefix, snap_min, snap_max, cols, spin_sampler,
        primary_spin=False, rstate=None, progress_bar=True, kwargs_tqdm = {}
    ):
        kwargs_tqdm.update(desc = 'Adding spin parameters')
        super().__init__(
            handler = handler,
            hc_path = hc_path,
            hc_prefix = hc_prefix,
            snap_min = snap_min,
            snap_max = snap_max,
            key_new = 'Spin',
            col_inh = [cols['spin'], cols['L_x'], cols['L_y'], cols['L_z']],
            col_req = ['ProgIndex', 'ProgNumber', cols['m_vir'], cols['r_vir'], cols['v_vir']],
            read_prog = ['ID', cols['m_vir']],
            inherit = True,
            forward = True,
            progress_bar = progress_bar,
            kwargs_tqdm = kwargs_tqdm
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
        self.mass_conversion = handler.mass_conversion


class PlainSpinConstructor(Constructor):
    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        col_m_vir = self.col_m_vir
        col_r_vir = self.col_r_vir
        col_v_vir = self.col_v_vir
        spin_sampler = self.spin_sampler
        rstate = self.rstate
        mass_conversion = self.mass_conversion
        #
        df_req_mc = df_req[df_req.index < 0]
        spin = spin_sampler.Draw(len(df_req_mc), z)
        L = np.sqrt(2.)/mass_conversion*np.asarray(
            df_req_mc[col_m_vir]*df_req_mc[col_r_vir]*df_req_mc[col_v_vir]*spin
        )
        L = RandomUnitSphere(len(df_req_mc), rstate)*L
        df_spin = pd.DataFrame(
            np.vstack([spin, L]).T, index = df_req_mc.index, columns = df_inh.columns
        )
        df_spin = df_inh.append(df_spin)
        return df_spin


    def __init__(self,
        handler, hc_path, hc_prefix, snap_min, snap_max, cols, spin_sampler,
        rstate=None, progress_bar=True, kwargs_tqdm={}
    ):
        kwargs_tqdm.update(desc = 'Adding spin parameters')
        super().__init__(
            handler = handler,
            hc_path = hc_path,
            hc_prefix = hc_prefix,
            snap_min = snap_min,
            snap_max = snap_max,
            key_new = 'Spin',
            col_inh = [cols['spin'], cols['L_x'], cols['L_y'], cols['L_z']],
            col_req = ['ProgIndex', 'ProgNumber', cols['m_vir'], cols['r_vir'], cols['v_vir']],
            read_prog = [],
            inherit = True,
            forward = True,
            progress_bar = progress_bar,
            kwargs_tqdm = kwargs_tqdm
        )
        self.col_m_vir = cols['m_vir']
        self.col_r_vir = cols['r_vir']
        self.col_v_vir = cols['v_vir']
        self.spin_sampler = spin_sampler
        if rstate is None:
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate
        self.mass_conversion = handler.mass_conversion
