from .constructor import Constructor

import numpy as np
import pandas as pd
from tqdm import tqdm


class SpinConstructor(Constructor):
    def _InheritSpin(self, prog):
        pool = self.pool
        m_max = -1.
        spin = np.nan
        for p_id, _, m in prog:
            if m > m_max:
                m_max = m
                spin = pool[p_id]
            del pool[p_id]
        return spin


    def AddProperty(self, df_inh, df_req, list_prog, snap, z):
        # Load local variables
        pool = self.pool
        snap_min = self.snap_min
        spin_key = self.spin_key
        spin_sampler = self.spin_sampler
        #
        halo_id_mc = df_req.index[df_req.index < 0]
        spin = np.zeros(len(halo_id_mc))
        if snap > snap_min:
            prog_pointer = df_req.loc[df_req.index < 0, ['ProgIndex', 'ProgNumber']].values
            cond = prog_pointer[:, -1] > 0
            spin[cond] = [
                self._InheritSpin(list_prog[idx:idx + num]) for idx, num in prog_pointer[cond]
            ]
            cond = ~cond
        else:
            cond = np.full(len(spin), True)
        spin[cond] = spin_sampler.Draw(sum(cond), z)
        df_spin = pd.DataFrame(spin, index = halo_id_mc, columns = [spin_key])
        df_spin = df_inh.append(df_spin)
        # Walk forwad
        pool.update(
            {h_id:s for h_id, s in zip(df_req.index, df_spin.loc[df_req.index, spin_key])}
        )
        return df_spin


    def __init__(self,
        handler, hc_path, hc_prefix, snap_min, snap_max, hdf_key, spin_key, spin_sampler,
        progress_bar = True, kwargs_tqdm = {}
    ):
        kwargs_tqdm.update(desc = 'Adding spin parameters')
        super().__init__(
            handler = handler,
            hc_path = hc_path,
            hc_prefix = hc_prefix,
            snap_min = snap_min,
            snap_max = snap_max,
            key_new = hdf_key,
            col_inh = [spin_key],
            col_req = ['ProgIndex', 'ProgNumber'],
            inherit = True,
            read_prog = True,
            forward = True,
            progress_bar = progress_bar,
            kwargs_tqdm = kwargs_tqdm
        )
        self.pool = {}
        self.snap_min = snap_min
        self.spin_key = spin_key
        self.spin_sampler = spin_sampler
