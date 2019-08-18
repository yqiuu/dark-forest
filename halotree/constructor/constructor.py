import os
from abc import ABC, abstractmethod 

from ..utils import SnapRange
from ..connector import ColumnLookup

import numpy as np
import pandas as pd


class Constructor(ABC):
    def _PrepareReqData(self, store):
        # Load local variables
        column_lookup = dict(self.column_lookup)
        #
        df_req = pd.read_hdf(store, 'Host', columns = column_lookup['/Host'])
        del column_lookup['/Host']
        for key in column_lookup:
            df_tmp = pd.read_hdf(store, key, columns = column_lookup[key])
            df_req = pd.concat([df_req, df_tmp], axis =  1, join = 'inner')
        return df_req

    
    def _PrepareInhData(self, store, host_id_hc):
        # Load local variables
        handler = self.handler
        col_inh = self.col_inh
        inherit = self.inherit
        # 
        halo_id_sim = handler['ID'].astype(host_id_hc.dtype)
        df_sim = pd.DataFrame(index = halo_id_sim)
        if inherit:
            for col in col_inh:
                try:
                    df_sim[col] = handler[col]
                except KeyError:
                    pass
            try:
                halo_id_fix = pd.read_hdf(store, 'Forward', columns = []).index
                df_sim = df_sim.loc[halo_id_fix[halo_id_fix >= 0]]
            except KeyError:
                index_drop = np.setdiff1d(
                    halo_id_sim[handler['HostHaloID'] == handler.HostID()],
                    host_id_hc, assume_unique = True
                )
                df_sim.drop(index = index_drop, inplace = True)
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
            np.zeros([len(halo_id_mc), len(col_inh)]), index = halo_id_mc, columns = col_inh
        )
        df_new = df_inh.append(df_new)
        return df_new


    def Run(self):
        # Load local variables
        _snap_range = self._snap_range
        handler = self.handler
        hc_path = self.hc_path
        hc_prefix = self.hc_prefix
        key_new = self.key_new
        inherit = self.inherit
        read_prog = self.read_prog
        #
        redshifts = handler.redshifts
        for snap in _snap_range:
            z = redshifts[snap]
            handler.SwitchSnapshot(snap)
            with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap))) as store:
                if len(read_prog) > 0 and '/Prog' in store.keys():
                    df_prog = pd.read_hdf(store, 'Prog')
                    df_prog.reset_index(inplace = True)
                    if len(read_prog) == 1:
                        list_prog = list(df_prog[read_prog[0]])
                    else:
                        list_prog = list(df_prog[read_prog].to_records(index = False))
                else:
                    list_prog = None
                df_req = self._PrepareReqData(store)
                df_inh = self._PrepareInhData(store, df_req.index[df_req.index >= 0])
                df_new = self._AddProperty(df_inh, df_req, list_prog, snap, z)
                df_new.to_hdf(store, key_new, format = 'table')


    def __init__(self,
        handler, hc_path, hc_prefix, snap_min, snap_max, key_new, col_inh, col_req,
        read_prog = [], inherit = True, forward = True, progress_bar = True, kwargs_tqdm = {}
    ):  
        self.handler = handler
        self.hc_path = hc_path
        self.hc_prefix = hc_prefix
        self.key_new = key_new
        self.col_inh = list(col_inh)
        self.col_req = list(col_req)
        self.inherit = inherit
        self.read_prog = read_prog
        with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap_min))) as store:
            self.column_lookup = ColumnLookup(store, col_req)
        if forward:
            snap_range = SnapRange(
                snap_min, snap_max + 1, progress_bar = progress_bar, **kwargs_tqdm
            )
        else:
            snap_range = SnapRange(
                snap_max, snap_min - 1, -1, progress_bar = progress_bar, **kwargs_tqdm
            )
        self._snap_range = snap_range


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
