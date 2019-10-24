import os
from abc import ABC, abstractmethod 

from ..connector import ColumnLookup

import numpy as np
import pandas as pd
from tables import NoSuchNodeError
from tqdm import trange


class Constructor(ABC):
    def _CreateColumnName(self, col_req, snap_min):
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


    def _SnapRange(self):
        # Load local variables
        snap_max = self.snap_max
        snap_min = self.snap_min
        forward = self.forward
        desc_tqdm = self.desc_tqdm
        kwargs_tqdm = self.kwargs_tqdm
        #
        kwargs_tqdm.update(desc=desc_tqdm)
        if self.forward:
            return trange(snap_min, snap_max + 1, **kwargs_tqdm)
        else:
            return trange(snap_max, snap_min - 1, -1, **kwargs_tqdm)


    def _PrepareProgData(self):
        cntr_target = self.cntr_target
        read_prog = self.read_prog
        if len(read_prog) > 0:
            try:
                rec_prog = cntr_target.Read('Prog', read_prog, 'records')
            except NoSuchNodeError:
                return None
            if len(read_prog) == 1:
                list_prog = list(rec_prog[read_prog[0]])
            return list_prog
        else:
            return None


    def _PrepareReqData(self):
        # Load local variables
        cntr_target = self.cntr_target
        col_host = self.col_host
        col_main = self.col_main
        #
        df_host = cntr_target.Read('Host', col_host, 'pandas', 'Index')
        if len(col_main) > 0:
            df_main = cntr_target.Read('Main', col_main, 'pandas', 'Index')
            df_req = pd.concat([df_host, df_main], axis=1, join='inner')
        else:
            df_req = df_host
        return df_req

    
    def _PrepareInhData(self, host_id_hc):
        # Load local variables
        cntr_target = self.cntr_target
        cntr_source = self.cntr_source
        col_inh = self.col_inh
        inherit = self.inherit
        # 
        halo_id_sim = cntr_source['ID'].astype(host_id_hc.dtype)
        df_sim = pd.DataFrame(index=halo_id_sim)
        if inherit:
            for col in col_inh:
                try:
                    df_sim[col] = cntr_source[col]
                except KeyError:
                    pass
            index_drop = np.setdiff1d(
                halo_id_sim[cntr_source['HostHaloID'] == cntr_source.HostID()],
                host_id_hc, assume_unique=True
            )
            df_sim.drop(index=index_drop, inplace=True)
        return df_sim


    @abstractmethod
    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        """
        Parameters
        ----------
            df_inh: pandas.DataFrame:
                Requested data read from the handler.

                column:
                    If `inherit` = True, columns are specified by `col_inh`.
                    If 'inherit` = False, there will be no column.
                index:
                    Halo ID of all simulation host halos stored in the halo
                    connector and all sub halos read from the handler.
            df_req: pandas.DataFrame
                Requested data read from the halo connector.

                column:
                    Specified by `col_req`.
                index:
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


    def _SaveOutput(self, df_new, snap):
        hf = self.cntr_target.hf
        try:
            group = hf.get_node('/Snapshot_%03d/Main'%snap)
        except NoSuchNodeError:
            group = hf.create_group('/Snapshot_%03d'%snap, 'Main')
        for col in df_new.columns:
            hf.create_carray(group, col, obj=df_new[col].values)


    def Run(self):
        # Load local variables
        cntr_target = self.cntr_target
        cntr_source = self.cntr_source
        inherit = self.inherit
        read_prog = self.read_prog
        #
        redshifts = cntr_source.redshifts
        for snap in self._SnapRange():
            z = redshifts[snap]
            cntr_target.SwitchSnapshot(snap)
            cntr_source.SwitchSnapshot(snap)
            df_req = self._PrepareReqData()
            df_inh = self._PrepareInhData(df_req.index[df_req.index >= 0])
            list_prog = self._PrepareProgData()
            df_new = self._AddProperty(df_inh, df_req, list_prog, snap, z)
            self._SaveOutput(df_new, snap)


    def __init__(self,
        cntr_target, cntr_source, col_inh, col_req, snap_max, snap_min,
        read_prog, inherit, forward, desc_tqdm, kwargs_tqdm
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
        self.desc_tqdm = desc_tqdm
        self.kwargs_tqdm = kwargs_tqdm
        self._CreateColumnName(col_req, snap_min)


class QuickConstructor(Constructor):
    def _AddProperty(self, df_inh, df_req):
        df_req = df_req[df_req.index < 0]
        df_quick = pd.DataFrame(
            self.transform(df_req.values), index = df_req.index, columns = df_inh.columns
        )
        df_quick = df_inh.append(df_quick)
        return df_quick


    def Run(self, snap):
        # Load local variables
        handler = self.handler
        hc_path = self.hc_path
        hc_prefix = self.hc_prefix
        col_req = self.col_req
        #
        handler.SwitchSnapshot(snap)
        with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap))) as store:
            try:
                self.column_lookup
            except AttributeError:
                 self.column_lookup = ColumnLookup(store, col_req)
            df_req = self._PrepareReqData(store)
            df_inh = self._PrepareInhData(store, df_req.index[df_req.index >= 0])
            df_new = self._AddProperty(df_inh, df_req)
        return df_new


    def __init__(self, handler, hc_path, hc_prefix, col_inh, col_req, transform):
        self.handler = handler
        self.hc_path = hc_path
        self.hc_prefix = hc_prefix
        self.col_inh = [col_inh]
        self.col_req = [col_req]
        self.transform = transform
        self.inherit = True
