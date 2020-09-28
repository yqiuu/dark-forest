from .constructor import Constructor

import numpy as np
import pandas as pd


__all__ = ['EntropyConstructor']


class EntropyConstructor(Constructor):
    def __init__(self,
        cols, snap_max, snap_min=0, postfix=None, nc=2, beta=.75, gamma=1./3.,
        kwargs_tqdm={}
    ):
        col_req= [
            'ID', 'DescendantSnap', 'Descendant', 'Primary', cols['m_main']
        ]
        super().__init__(
            col_inh=[],
            col_req=col_req,
            snap_max=snap_max,
            snap_min=snap_min,
            read_prog=[],
            inherit=True,
            forward=True,
            postfix=postfix,
            message='Adding tree entropy parameters',
            kwargs_tqdm=kwargs_tqdm
        )
        self.col_m_vir = cols['m_main']
        self.p_alpha = 1/np.log(nc) + 1
        self.p_f = (self.p_alpha - 1)*np.e
        self.p_a = (2 - gamma)/self.p_f
        self.p_b = nc*(1 - beta) - 1 - self.p_a
        self.p_gamma = gamma


    def run(self, cntr_ext, cntr_src):
        self.pool = [{} for snap in range(self.snap_max + 1)]
        super().run(cntr_ext, cntr_src)


    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        # Entropy paramters
        alpha = self.p_alpha
        f = self.p_f
        a = self.p_a
        b = self.p_b
        gamma = self.p_gamma
        #
        pool = self.pool
        df_req = df_req[df_req.index < 0]
        entropy = np.zeros(len(df_req), dtype='f8')
        if snap != self.snap_min:
            pool_snap = pool[snap]
            data = [
                df_req['ID'].values,
                df_req[self.col_m_vir].values,
                df_req['Primary'].values,
            ]
            for i_halo, (id_h, m_h, pr_h) in enumerate(zip(*data)): 
                if id_h in pool_snap:
                    m_prog, s_prog = np.asarray(pool_snap[id_h]).T
                    m_tot = m_prog.sum()
                    x_prog = m_prog/m_tot
                    H = -f*np.sum(x_prog**alpha*np.log(x_prog))
                    w = 1 + a*H + b*H*H
                    s_new = H + w*np.sum(x_prog*x_prog*(s_prog - H))
                    if pr_h:
                        s_new = min(1., s_new*(m_tot/m_h)**gamma)
                    entropy[i_halo] = s_new
                else:
                    entropy[i_halo] = 0.
        df_entropy = pd.DataFrame(
            entropy, index=df_req.index, columns=['TreeEntropy']
        )
        #
        data = [
            df_req['DescendantSnap'].values,
            df_req['Descendant'].values,
            df_req[self.col_m_vir].values,
            entropy
        ]
        for snap_d, id_d, m_d, en_d in zip(*data):
            if id_d in pool[snap_d]:
                pool[snap_d][id_d].append((m_d, en_d))
            else:
                pool[snap_d][id_d] = [(m_d, en_d)]
        return df_entropy

