import os

from ..utils import Range
from ..connector import RemoveKey, ColumnLookup
from .constructor import Constructor

import numpy as np
import pandas as pd


class ForwardRebuilder(Constructor):
    # [1] Assume that it points to itself when there is no descendant halo
    def _RemoveLostSubhalo(self, df_sim):
        halo_id_sim = df_sim.index.values
        host_id_all = df_sim['HostHaloID'].values
        select_sub_halo = host_id_all != self.handler.HostID()
        halo_id_sub = halo_id_sim[select_sub_halo]
        host_id_sub = host_id_all[select_sub_halo]
        halo_id_host = halo_id_sim[~select_sub_halo]
        df_sim.drop(
            index = halo_id_sub[np.in1d(host_id_sub, halo_id_host, invert = True)], inplace = True
        )


    def _FixSubhaloDescendant(self, df_target, pool_surv):
        df_sub = df_target[df_target['HostHaloID'] != self.handler.HostID()]
        if len(df_sub > 0):
            halo_id_sub = [
                s_id for s_id, _, desc_snap, desc_id, _ in df_sub.itertuples() \
                    if desc_id not in pool_surv[desc_snap]
            ]
            df_target.loc[halo_id_sub, ['DescendantSnap', 'Descendant']] = df_target.loc[
                df_target.loc[halo_id_sub, 'HostHaloID'], ['DescendantSnap', 'Descendant']
            ].values

    
    def _CutDescendant(self, df_sim, snap):
        select_cut = df_sim['DescendantSnap'] > self.snap_max
        df_sim.loc[select_cut, 'DescendantSnap'] = snap
        df_sim.loc[select_cut, 'Descendant'] = df_sim.index[select_cut] # [1]


    def _ReubildOriginalTree(self):
        # Load local variables
        handler = self.handler
        hc_path = self.hc_path
        hc_prefix = self.hc_prefix
        snap_min = self.snap_min
        snap_max = self.snap_max
        _snap_range = self._snap_range
        #
        n_snap = snap_max + 1
        list_df_surv = [None]*n_snap
        pool_surv = [None]*n_snap
        snap_target = [None]*n_snap
        for snap in _snap_range:
            handler.SwitchSnapshot(snap)
            with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap))) as store:
                halo_id = pd.read_hdf(store, 'Host', columns = []).index
                df_sim = self._PrepareInhData(store, halo_id[halo_id >= 0])
            self._RemoveLostSubhalo(df_sim)
            if len(df_sim) > 0:
                self._CutDescendant(df_sim, snap)
                snap_target[snap] = max(df_sim['DescendantSnap'])
            else:
                snap_target[snap] = snap_min - 1
            list_df_surv[snap] = df_sim
            pool_surv[snap] = {h_id:0 for h_id in df_sim.index.values}
            for snap_t in range(snap_min, snap):
                if snap == snap_target[snap_t]:
                    df_target = list_df_surv[snap_t]
                    self._FixSubhaloDescendant(df_target, pool_surv)
                    df_target.to_hdf(
                        os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap_t)), '_ForwardFix'
                    )
                    # Release memory
                    list_df_surv[snap_t] = None
                    pool_surv[snap_t] = None
            if snap == snap_max:
                df_sim.to_hdf(
                    os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap)), '_ForwardFix'
                )
        self.snap_jump = max(snap_t - snap for snap_t, snap in zip(snap_target, range(n_snap)))


    def _PrepareReqData(self, store):
        if '/_ForwardFix' in store.keys():
            self.df_forward_fix = pd.read_hdf(store, '_ForwardFix')
        else:
            self.df_forward_fix = pd.DataFrame()
        return pd.read_hdf(store, 'Host', columns = ['ProgIndex', 'ProgNumber'])


    def _GeneratePointerMC(self, df_inh, df_req, list_prog, snap):
        # Load local variables
        handler = self.handler
        snap_max = self.snap_max
        forest_id_max = self.forest_id_max
        df_forward_next = self.df_forward_next
        #
        select_mc_halo = df_req.index < 0
        df_forward_mc = pd.DataFrame(index = df_req.index[select_mc_halo])
        n_halo_mc = len(df_forward_mc)
        n_halo_sim = len(df_inh)
        df_forward_mc['ID'] = handler.NewID(snap, np.arange(n_halo_sim, n_halo_sim + n_halo_mc))
        df_forward_mc['HostHaloID'] = handler.HostID(n_halo_mc)
        if snap == snap_max:
            df_forward_mc['DescendantSnap'] = snap
            df_forward_mc['Descendant'] = df_forward_mc['ID'] # [1]
            df_forward_mc['ForestID'] = np.arange(forest_id_max, forest_id_max + n_halo_mc)
        else:
            df_forward_mc = df_forward_mc.join(df_forward_next)
        # Put DescendantSnap/Descendant/ForestID into the pool
        progenitors = {}
        halo_id_host_sim = df_req.index[~select_mc_halo]
        if n_halo_sim > 0:
            df_forward = df_inh.loc[df_req.index[~select_mc_halo]]
            df_forward = df_forward.append(df_forward_mc)
            df_forward = df_forward.loc[df_req.index]
        else:
            df_forward = df_forward_mc
        cond = df_req['ProgNumber'] > 0
        for (idx, num), (h_id, f_id) in zip(
            df_req.loc[cond, ['ProgIndex', 'ProgNumber']].values,
            df_forward.loc[cond, ['ID', 'ForestID']].values
        ):
            for p_id in list_prog[idx:idx + num]:
                if p_id < 0:
                    progenitors[p_id] = (snap, h_id, f_id)
        self.df_forward_next = pd.DataFrame.from_dict(
            progenitors, orient = 'index', columns = ['DescendantSnap', 'Descendant', 'ForestID']
        )
        return df_forward_mc


    def _GeneratePointerSim(self, df_inh, snap):
        """
        Generate new IDs for simulation halos, and change Descendant/HostHaloID
        accordingly.
        """
        # Load local variables
        handler = self.handler
        pool_sim = self.pool_sim
        snap_max = self.snap_max
        snap_jump = self.snap_jump
        #
        df_inh.insert(0, 'ID', handler.NewID(snap, np.arange(len(df_inh))))
        #
        pool_sim[snap] = {old_id:new_id for old_id, new_id in zip(df_inh.index, df_inh['ID'])}
        # Fix host halo IDs
        select_sub_halo = df_inh['HostHaloID'] != handler.HostID()
        halo_id_sub = df_inh.loc[df_inh.loc[select_sub_halo, 'HostHaloID'], 'ID'].values
        df_inh.loc[select_sub_halo, 'HostHaloID'] = halo_id_sub
        # Generate new Descendant for simulations halos
        df_inh['Descendant'] = list(map(
            lambda desc_snap, desc_id:pool_sim[desc_snap][desc_id],
            *df_inh[['DescendantSnap', 'Descendant']].values.T
        ))
        # Clean memory
        snap_clean = snap + snap_jump
        if snap_clean <= snap_max:
            pool_sim[snap_clean] = None


    def _Clean(self):
        if not self.test_mode:
            RemoveKey(
                self.hc_path, self.hc_prefix, self.snap_min, self.snap_max,
                '_ForwardFix', self.kwargs_tqdm
            )


    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        df_inh = self.df_forward_fix
        if len(df_inh) > 0:
            self._GeneratePointerSim(df_inh, snap)
        df_forward = self._GeneratePointerMC(df_inh, df_req, list_prog, snap)
        df_forward = df_inh.append(df_forward)
        return df_forward

        
    def Run(self):
        try:
            # Load local variables
            snap_min = self.snap_min
            snap_max = self.snap_max
            kwargs_tqdm = self.kwargs_tqdm
            #
            self.col_inh = ['HostHaloID', 'DescendantSnap', 'Descendant', 'ForestID']
            self.inherit = True
            kwargs_tqdm.update(desc = 'Rebuilding original trees')
            self._snap_range = Range(snap_min, snap_max + 1, **kwargs_tqdm)
            self._ReubildOriginalTree()
            #
            self.col_inh = []
            self.inherit = False
            kwargs_tqdm.update(desc = 'Adding forward pointers')
            self._snap_range = Range(snap_max, snap_min - 1, -1, **kwargs_tqdm)
            super().Run()
        finally:
            self._Clean()


    def __init__(self,
        handler, hc_path, hc_prefix, snap_min, snap_max, kwargs_tqdm = {}
    ):  
        self.handler = handler
        self.hc_path = hc_path
        self.hc_prefix = hc_prefix
        self.snap_min = snap_min
        self.snap_max = snap_max
        self.kwargs_tqdm = kwargs_tqdm
        #
        self.key_new = 'Forward'
        self.read_prog = ['ID']
        self.pool_sim = [{} for snap in range(snap_max + 1)]
        self.df_forward_next = None
        self.forest_id_max = handler.MaxForestID(snap_max)
        #
        self.test_mode = False


def WalkForward(hc_path, hc_prefix, snap_min, snap_max, kwargs_tqdm = {}):
    #
    pool = [{} for snap in range(snap_max + 1)]
    #
    kwargs_tqdm.update(desc = 'Walking forward')
    for snap in Range(snap_min, snap_max + 1, **kwargs_tqdm):
        df_forward = pd.read_hdf(
            os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap)), 'Forward'
        )
        for desc_snap, desc_id in df_forward[['DescendantSnap', 'Descendant']].values:
            pool[desc_snap][desc_id] = 0
        if snap > snap_min:
            pool_curr = pool[snap]
            for halo_id in df_forward['ID']:
                if halo_id in pool_curr:
                    del pool_curr[halo_id]
    for snap, p in zip(range(snap_min, snap_max + 1), pool):
        if len(p) > 0:
            raise ValueError(
                'Find unwalkable halos at snapshot %d ('%snap + \
                    ', '.join(str(k) for k in p.keys()) + ')'
            )
