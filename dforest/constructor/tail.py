from .constructor import Constructor

import numpy as np
import pandas as pd


__all__ = ["TailConstructor"]


class TailConstructor(Constructor):
    """Add primary progenitor pointers, using self-links for halos without progenitors."""

    def __init__(
        self, cntr_ext, cntr_src, cols, snap_max, snap_min=0, postfix=None, kwargs_tqdm={}
    ):
        super().__init__(
            cntr_ext=cntr_ext,
            cntr_src=cntr_src,
            col_inh=[cols["tail"]],
            col_req=["ProgIndex", "ProgNumber"],
            snap_min=snap_min,
            snap_max=snap_max,
            read_prog=["Index", cols["m_main"]],
            inherit=False,
            forward=True,
            postfix=postfix,
            message="Adding primary progenitor pointers",
            kwargs_tqdm=kwargs_tqdm,
        )
        self.col_tail = cols["tail"]

    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        col_tail = self.col_tail
        #
        if list_prog is None:
            select_mc = df_req.index < 0
            df_tail_mc = pd.DataFrame(index=df_req.index[select_mc])
            df_tail_mc[col_tail] = df_tail_mc.index
            df_tail_sim = pd.DataFrame(index=df_inh.index)
            df_tail_sim[col_tail] = df_tail_sim.index
            df_tail = df_tail_sim.append(df_tail_mc)
            return df_tail
        #
        df_tail = pd.DataFrame(
            np.full(len(df_req), -1, "i8"), index=df_req.index, columns=[col_tail]
        )
        select_sim = df_tail.index > 0
        df_tail.loc[select_sim, col_tail] = df_inh.loc[df_tail.index[select_sim], col_tail]
        cond = df_req["ProgNumber"] == 0
        df_tail.loc[cond, col_tail] = df_req.index[cond]
        #
        list_tail = list(df_tail[col_tail].values)
        for i_halo, (idx, num) in enumerate(df_req.values):
            if num == 0:
                continue

            tail_sim = list_tail[i_halo]
            prog = list_prog[idx : idx + num]

            tail, m_max = prog[0]
            for id_p, m_p in prog:
                if id_p == tail_sim:
                    tail = tail_sim
                    break
                elif m_p > m_max:
                    m_max = m_p
                    tail = id_p
            list_tail[i_halo] = tail
        df_tail[col_tail] = list_tail
        df_inh.loc[df_tail.index[select_sim]] = df_tail[select_sim]
        df_inh.index.name = df_tail.index.name
        df_tail = df_inh.append(df_tail[~select_sim])
        return df_tail
