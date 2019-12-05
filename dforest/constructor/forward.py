from .constructor import Constructor

import numpy as np
import pandas as pd
from tqdm import trange
from tables import NoSuchNodeError


def PrepareSourceData(cntr_source):
    df_source = pd.DataFrame(index=cntr_source['ID'])
    df_source['HostHaloID'] = cntr_source['HostHaloID']
    df_source['DescendantSnap'] = cntr_source['DescendantSnap']
    df_source['Descendant'] = cntr_source['Descendant']
    df_source['ForestID'] = cntr_source['ForestID']
    return df_source


def PutTopHalo(cntr_target, cntr_source, pool_sub, df_source):
    halo_id_surv = cntr_target.Read('Host', ['Index'])['Index']
    halo_id_surv = halo_id_surv[halo_id_surv > 0]

    halo_id = df_source.index.values
    host_id = df_source['HostHaloID'].values
    select_host = host_id == cntr_source.HostID()
    host_id[select_host] = halo_id[select_host]
    select_top = np.in1d(halo_id, pool_sub.keys(), invert=True)
    for h_id, surv in zip(halo_id[select_top], np.in1d(host_id[select_top], halo_id_surv)):
        pool_sub[h_id] = surv


def SetSurviveIndex(pool_sub, df_source):
    df_source['Survive'] = 0
    df_source.loc[list(pool_sub.keys()), 'Survive'] = list(pool_sub.values())


def CutDescendant(df_source, snap, snap_max):
    # [1] Assuem that it points to itself when there is no descendant halo
    select_cut = df_source['DescendantSnap'] > snap_max
    df_source.loc[select_cut, 'DescendantSnap'] = snap
    df_source.loc[select_cut, 'Descendant'] = df_source.index[select_cut] # [1]


def WalkForward(pool, df_source):
    for desc_snap, desc_id, surv in df_source[['DescendantSnap', 'Descendant', 'Survive']].values:
        pool[desc_snap][desc_id] = surv


def FixSubhaloDescendant(cntr_source, df_source, pool):
    df_sub = df_source[df_source['HostHaloID'] != cntr_source.HostID()]
    if len(df_sub > 0):
        halo_id = df_sub.index.values
        desc_snap = df_sub['DescendantSnap'].values
        desc_id = df_sub['Descendant'].values
        halo_id_sub = [
            s_id for s_id, desc_snap, desc_id in zip(halo_id, desc_snap, desc_id) \
                if not pool[desc_snap][desc_id]
        ]
        host_id = df_source.loc[halo_id_sub, 'HostHaloID'].values
        if len(host_id) > 0:
            df_source.loc[halo_id_sub, 'DescendantSnap'] \
                = df_source.loc[host_id, 'DescendantSnap'].values
            df_source.loc[halo_id_sub, 'Descendant'] \
                = df_source.loc[host_id, 'Descendant'].values


def SaveOutput(cntr_target, df_source, snap):
    df_save = df_source[df_source['Survive']]
    if len(df_save):
        hf = cntr_target.hf
        group = hf.create_group('/Snapshot_%03d'%snap, '_ForwardFix')
        hf.create_carray(group, 'Index', obj=df_save.index.values.astype('i8'))
        hf.create_carray(group, 'HostHaloID', obj=df_save['HostHaloID'].values.astype('i8'))
        hf.create_carray(group, 'DescendantSnap', obj=df_save['DescendantSnap'].values.astype('i4'))
        hf.create_carray(group, 'Descendant', obj=df_save['Descendant'].values.astype('i8'))
        hf.create_carray(group, 'ForestID', obj=df_save['ForestID'].values.astype('i8'))


def RebuildForwardConnection(cntr_target, cntr_source, snap_max, snap_min=0, kwargs_tqdm={}):
    n_snap = snap_max + 1
    df_source_all = [None]*n_snap
    pool = [{} for i_snap in range(n_snap)]
    snap_target = [None]*n_snap
    kwargs_tqdm.update(desc="Rebuilding original trees")
    for snap in trange(snap_min, n_snap, **kwargs_tqdm):
        cntr_target.SwitchSnapshot(snap)
        cntr_source.SwitchSnapshot(snap)

        pool_sub = pool[snap]
        df_source = PrepareSourceData(cntr_source)
        PutTopHalo(cntr_target, cntr_source, pool_sub, df_source)
        SetSurviveIndex(pool_sub, df_source)
        CutDescendant(df_source, snap, snap_max)
        WalkForward(pool, df_source)
        if len(df_source) > 0:
            snap_target[snap] = max(df_source['DescendantSnap'])
        else:
            snap_target[snap] = snap_min - 1
        df_source_all[snap] = df_source
        for snap_t in range(snap_min, snap):
            snap_target_it = snap_target[snap_t]
            if snap == snap_target_it:
                df_fix = df_source_all[snap_t]
                FixSubhaloDescendant(cntr_source, df_fix, pool)
                SaveOutput(cntr_target, df_fix, snap_t)
                # Clear memory
                if snap_t == snap_min or snap_target_it > max(snap_target[snap_min:snap_t]):
                    for snap_c in range(snap_min, snap_t + 1):
                        df_source_all[snap_c] = None
                        pool[snap_c] = None
        if snap == snap_max:
            SaveOutput(cntr_target, df_source, snap)
    snap_jump = max(snap_t - snap for snap_t, snap in zip(snap_target, range(n_snap)))
    return snap_jump


class ForwardRebuilder(Constructor):
    def _PrepareReqData(self):
        return self.cntr_target.Read('Host', ['ProgIndex', 'ProgNumber'], 'pandas', 'Index')


    def _PrepareInhData(self):
        try:
            # [1] Specifiy columns to obtain correct order
            cols = ['HostHaloID', 'DescendantSnap', 'Descendant', 'ForestID'] # [1]
            df_inh = self.cntr_target.Read('_ForwardFix', cols, 'pandas', 'Index')
        except NoSuchNodeError:
            df_inh = pd.DataFrame()
        return df_inh


    def _GeneratePointerMC(self, df_inh, df_req, list_prog, snap):
        # Load local variables
        cntr_source = self.cntr_source
        snap_max = self.snap_max
        forest_id_max = self.forest_id_max
        df_forward_next = self.df_forward_next
        #
        select_mc_halo = df_req.index < 0
        df_forward_mc = pd.DataFrame(index=df_req.index[select_mc_halo])
        n_halo_mc = len(df_forward_mc)
        n_halo_sim = len(df_inh)
        df_forward_mc['ID'] = cntr_source.NewID(snap, np.arange(n_halo_sim, n_halo_sim + n_halo_mc))
        df_forward_mc['HostHaloID'] = cntr_source.HostID(n_halo_mc)
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
            progenitors, orient='index', columns=['DescendantSnap', 'Descendant', 'ForestID']
        )
        return df_forward_mc


    def _GeneratePointerSim(self, df_inh, snap):
        """
        Generate new IDs for simulation halos, and change Descendant/HostHaloID
        accordingly.
        """
        # Load local variables
        cntr_source = self.cntr_source
        pool_sim = self.pool_sim
        snap_max = self.snap_max
        snap_jump = self.snap_jump
        #
        df_inh.insert(0, 'ID', cntr_source.NewID(snap, np.arange(len(df_inh))))
        #
        pool_sim[snap] = {old_id: new_id for old_id, new_id in zip(df_inh.index, df_inh['ID'])}
        # Fix host halo IDs
        select_sub_halo = df_inh['HostHaloID'] != cntr_source.HostID()
        halo_id_sub = df_inh.loc[df_inh.loc[select_sub_halo, 'HostHaloID'], 'ID'].values
        df_inh.loc[select_sub_halo, 'HostHaloID'] = halo_id_sub
        # Generate new Descendant for simulations halos
        desc_snap = df_inh['DescendantSnap'].values
        desc_id = df_inh['Descendant'].values
        df_inh['Descendant'] = list(map(
            lambda d_snap, d_id: pool_sim[d_snap][d_id], desc_snap, desc_id
        ))
        # Clean memory
        snap_clean = snap + snap_jump
        if snap_clean <= snap_max:
            pool_sim[snap_clean] = None


    def _Clean(self):
        if not self.test_mode:
            self.cntr_target.Remove('_ForwardFix')


    def _AddProperty(self, df_inh, df_req, list_prog, snap, z):
        if len(df_inh) > 0:
            self._GeneratePointerSim(df_inh, snap)
        df_forward = self._GeneratePointerMC(df_inh, df_req, list_prog, snap)
        df_forward = df_inh.append(df_forward)
        return df_forward


    def _SaveOutput(self, df_new, snap):
        hf = self.cntr_target.hf
        group = hf.create_group('/Snapshot_%03d'%snap, 'Main')
        for col in df_new.columns:
            hf.create_carray(group, col, obj=df_new[col].values)
        hf.create_carray(group, 'Index', obj=df_new.index.values)


    def Run(self):
        self.snap_jump = RebuildForwardConnection(
            self.cntr_target, self.cntr_source, self.snap_max, self.snap_min, self.kwargs_tqdm
        )
        super().Run()


    def __init__(self, cntr_target, cntr_source, snap_max, snap_min=0, kwargs_tqdm={}):
        super().__init__(
            cntr_target=cntr_target,
            cntr_source=cntr_source,
            col_inh=[],
            col_req=[],
            snap_max=snap_max,
            snap_min=snap_min,
            read_prog=['Index'],
            inherit=False,
            forward=False,
            desc_tqdm='Adding forward pointers',
            kwargs_tqdm=kwargs_tqdm
        )
        self.kwargs_tqdm = kwargs_tqdm
        self.pool_sim = [{} for snap in range(snap_max + 1)]
        self.df_forward_next = None
        self.forest_id_max = cntr_source.MaxForestID(snap_max)
        #
        self.test_mode = False
