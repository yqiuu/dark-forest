import numpy as np
import pandas as pd
from tqdm import trange


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
    hf = cntr_target.hf
    group = hf.create_group('/Snapshot_%03d'%snap, '_ForwardFix')
    hf.create_carray(group, 'Index', obj=df_save.index.values)
    hf.create_carray(group, 'HostHaloID', obj=df_save['HostHaloID'].values)
    hf.create_carray(group, 'DescendantSnap', obj=df_save['DescendantSnap'].values.astype('i4'))
    hf.create_carray(group, 'Descendant', obj=df_save['Descendant'].values)


def RebuildForwardConnection(cntr_target, cntr_source, snap_max, snap_min=0):
    n_snap = snap_max + 1
    df_source_all = [None]*n_snap
    pool = [{} for i_snap in range(n_snap)]
    snap_target = [None]*n_snap
    for snap in trange(snap_min, n_snap, desc="Rebuilding forward pointers"):
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
            if snap == snap_target[snap_t]:
                FixSubhaloDescendant(cntr_source, df_source, pool)
                SaveOutput(cntr_target, df_source, snap_t)
                df_source_all[snap_t] = None
                pool[snap_t] = None
        if snap == snap_max:
            SaveOutput(cntr_target, df_source, snap)
