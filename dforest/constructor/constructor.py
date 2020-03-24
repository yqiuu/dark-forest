import os
import gc
from abc import ABC, abstractmethod 

from ..logger import logger, dividing_line, memory_usage

import numpy as np
import pandas as pd
from tables import NoSuchNodeError
from tqdm import trange


__all__ = ['Constructor']


class Constructor(ABC):
    def __init__(self,
        cntr_target, cntr_source, col_inh, col_req, snap_max, snap_min,
        read_prog, inherit, forward, postfix, message, kwargs_tqdm
    ):
        self.cntr_target = cntr_target
        self.cntr_source = cntr_source
        self.col_inh = list(col_inh)
        self.col_req = list(col_req)
        self.snap_max = snap_max
        self.snap_min = snap_min
        self.read_prog = read_prog
        self.inherit = inherit
        self.forward = forward
        self.postfix = postfix
        self.message = message
        self.kwargs_tqdm = kwargs_tqdm
        self._set_column_names(col_req, snap_min)


    def run(self):
        # Load local variables
        cntr_target = self.cntr_target
        cntr_source = self.cntr_source
        inherit = self.inherit
        read_prog = self.read_prog
        #
        redshifts = cntr_source.redshifts
        n_snap = self.snap_max - self.snap_min + 1
        for i_snap, snap in enumerate(self._range()):
            logger.info(f"{self.message}: {i_snap+1}/{n_snap} ({memory_usage()})")
            z = redshifts[snap]
            df_req = self._prepare_req_data(snap)
            df_inh = self._prepare_inh_data(snap)
            df_inh.index = df_inh.index.astype(df_req.index.dtype)
            list_prog = self._prepare_prog_data(snap)
            df_new = self._add_property(df_inh, df_req, list_prog, snap, z)
            self._save_output(df_new, snap)
            gc.collect()
            logger.info(dividing_line())


    def _set_column_names(self, col_req, snap_min):
        node_host = [
            node.name for node in self.cntr_target.hf.iter_nodes('/Snapshot_%03d/Host'%snap_min)
        ]
        col_host = []
        col_main = []
        for c in col_req:
            if c in node_host:
                col_host.append(c)
            else:
                col_main.append(c)
        self.col_host = col_host
        self.col_main = col_main


    def _range(self):
        kwargs_tqdm = self.kwargs_tqdm
        kwargs_tqdm.update(desc=self.message)
        if self.forward:
            return trange(self.snap_min, self.snap_max + 1, **kwargs_tqdm)
        else:
            return trange(self.snap_max, self.snap_min - 1, -1, **kwargs_tqdm)


    def _prepare_prog_data(self, snap):
        cntr_target = self.cntr_target
        read_prog = self.read_prog
        if len(read_prog) > 0:
            try:
                rec_prog = cntr_target.read_cols(snap, 'Prog', read_prog, 'records')
            except OSError:
                return None
            if len(read_prog) == 1:
                list_prog = list(rec_prog[read_prog[0]])
            else:
                list_prog = rec_prog
            return list_prog
        else:
            return None


    def _prepare_req_data(self, snap):
        # Load local variables
        cntr_target = self.cntr_target
        col_host = self.col_host
        col_main = self.col_main
        #
        df_host = cntr_target.read_cols(snap, 'Host', col_host, 'pandas', 'Index')
        if len(col_main) > 0:
            df_main = cntr_target.read_cols(snap, 'MainA', col_main, 'pandas', 'Index')
            df_req = pd.concat([df_host, df_main], axis=1, join='inner')
        else:
            df_req = df_host
        return df_req


    def _prepare_inh_data(self, snap):
        # Load local variables
        cntr_target = self.cntr_target
        cntr_source = self.cntr_source
        col_inh = self.col_inh
        # 
        halo_id_sim = cntr_source[snap, 'ID']
        df_sim = pd.DataFrame(index=halo_id_sim)
        for col in col_inh:
            try:
                df_sim[col] = cntr_source[snap, col]
            except KeyError:
                pass
        try:
            halo_id_fix = cntr_target.read_cols(snap, 'MainA', ['Index'])['Index']
        except IOError:
            raise Exception("run FowardRebuilder before adding any other properties") from None
        df_sim = df_sim.loc[halo_id_fix[halo_id_fix > 0]]
        return df_sim


    @abstractmethod
    def _add_property(self, df_inh, df_req, list_prog, snap, z):
        """
        Parameters
        ----------
        df_inh: pandas.DataFrame:
            Requested data read from the handler.
            Column:
                Specified by `col_inh`.
            Index:
                Halo ID of all simulation host halos stored in the halo
                connector and all sub halos read from the handler.
        df_req: pandas.DataFrame
            Requested data read from the halo connector.
            Column:
                Specified by `col_req`.
            Index:
                The same with those stored in the halo connector.
        list_prog: list
            Progenitor data read from the halo connector.
        snap: int
            Snapshot number that the data should be read at.
        z: float
            Redshift of the given snapshot.
        """

        # Load local variables
        col_inh = self.col_inh
        #
        halo_id_mc = df_req.index[df_req.index < 0]
        df_new = pd.DataFrame(
            np.zeros([len(halo_id_mc), len(col_inh)]), index=halo_id_mc, columns=col_inh
        )
        df_new = df_inh.append(df_new)
        return df_new


    def _save_output(self, df_new, snap):
        hf = self.cntr_target.hf
        postfix = '' if self.postfix is None else f'_{self.postfix}'
        name = 'MainB' if self.inherit else 'MainA'
        group = hf.get_node(f'/Snapshot_{snap:03d}', name)
        try:
            inds = self.cntr_target.read_cols(snap, name, ['Index'])['Index']
        except NoSuchNodeError:
            inds = None
        if inds is None or np.array_equal(inds, df_new.index):
            for col in df_new.columns:
                col_save = f'{col}{postfix}'
                try:
                    hf.remove_node(group, col_save)
                except NoSuchNodeError:
                    pass
                hf.create_carray(group, col_save, obj=df_new[col].values)
        else:
            raise ValueError("Indices mismatch.")
