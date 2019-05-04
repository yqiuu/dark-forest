import os

from .logger import CreateLogger

import numpy as np
import pandas as pd
import h5py

__all__ = [
    'DataReader',
    'VELOCIraptorReader',
    'ConstructHaloConnector'
]


logger = CreateLogger(__name__)


class DataReader:
    def ReadSnapshot(self, snap):
        pass


    def __getitem__(self, key):
        pass

    
    def __enter__(self):
        pass


    def __exit__(self, *args):
        pass


    def __init__(self, fp):
        pass


class VELOCIraptorReader(DataReader):
    def ReadSnapshot(self, snap):
        self.dataset = self.fp.get('Snap_%03d'%snap)


    def __getitem__(self, key):
        if key == 'DescendantSnap':
            return np.asarray(self.dataset['Descendant'])//10**12
        else:
            return np.asarray(self.dataset[key])


    def __enter__(self):
        self.fp = h5py.File(self.fname, "r")
        return self

    
    def __exit__(self, *args):
        self.fp.close()
        

    def __init__(self, fname):
        self.fname = fname


def SaveOutput(data, columns, snap, path, prefix):
    plain_data = []
    count = np.zeros(len(data), dtype = int)
    for i_prog, progenitors in enumerate(data.values()):
        plain_data.extend(progenitors)
        count[i_prog] = len(progenitors)
    df_prog = pd.DataFrame(plain_data, columns = columns)
    df_prog.to_hdf(os.path.join(path, '%s_prog_%d.hdf5'%(prefix, snap)), "w")
    df_desc = pd.Series(np.append(0, np.cumsum(count)[:-1]), index = list(data.keys()))
    df_desc.to_hdf(os.path.join(path, '%s_host_%d.hdf5'%(prefix, snap)), "w")


def ConstructHaloConnector(reader, snap_start, snap_end, keys, path = './', prefix = 'hc'):
    columns = ['Progenitor', 'ProgenitorSnap', 'ProgenitorIndex']
    columns.extend(keys)
    #
    n_snap = snap_end + 1
    desc_pool_arr = np.asarray([{} for snap in range(n_snap)])
    for snap in range(snap_start, n_snap):
        logger.info("Processing snaphost %d"%snap)
        reader.ReadSnapshot(snap)
        # Merge subhalos to their host halo
        desc_pool = desc_pool_arr[snap]
        halo_id_arr = reader['ID']
        host_id_arr = reader['hostHaloID']
        host_table = dict(zip(halo_id_arr, host_id_arr))
        for halo_id in list(desc_pool.keys()):
            host_id = host_table[halo_id]
            progenitors = desc_pool[halo_id]
            if host_id != -1:
                try:
                    desc_pool[host_id].extend(progenitors)
                except KeyError:
                    desc_pool[host_id] = progenitors
                del desc_pool[halo_id]
        # Save existing descendants
        n_gal = len(halo_id_arr)
        snap_link = np.full(n_gal, -1, dtype = int)
        idx_link = np.full(n_gal, -1, dtype = int)
        cond = [True if halo_id in desc_pool else False for halo_id in halo_id_arr]
        n_save = np.sum(cond)
        if n_save > 0:
            snap_link[cond] = snap
            idx_link[cond] = np.arange(n_save)
            SaveOutput(desc_pool, columns, snap, path, prefix)
        # Walk forward
        desc_id_arr = reader['Descendant']
        desc_snap_arr = reader['DescendantSnap']
        props_arr = np.asarray([reader[k] for k in keys])
        for desc_id, desc_snap, host_id, halo_id, halo_snap, halo_idx, *props in zip(
            desc_id_arr, desc_snap_arr, host_id_arr, halo_id_arr, snap_link, idx_link, *props_arr
        ):
            if host_id == -1:
                target_pool = desc_pool_arr[desc_snap]
                try:
                    target_pool[desc_id].append((halo_id, halo_snap, halo_idx, *props))
                except KeyError:
                    target_pool[desc_id] = [(halo_id, halo_snap, halo_idx, *props)]

