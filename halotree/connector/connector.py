import os
from abc import ABC, abstractmethod

from ..utils import SnapRange
from ..logger import CreateLogger
from ..halotree import *

from tqdm import tqdm
import numpy as np
import pandas as pd

__all__ = [
    'Connector',
    'SaveHaloConnector',
    'ReadHaloConnector',
    'ColumnLookup',
    'RemoveKey'
]


logger = CreateLogger(__name__)


class Connector(ABC):
    @abstractmethod
    def SwitchSnapshot(self, snap):
        pass


    @abstractmethod
    def __getitem__(self, key):
        pass


    @abstractmethod
    def __enter__(self):
        pass


    @abstractmethod
    def __exit__(self, *args):
        pass


    @abstractmethod
    def __init__(self, fp):
        pass


def SaveOutput(handler, prog_data, mass_key, snap, path, prefix):
    mass_unit = handler.mass_conversion
    # Select all host halos at the current snapshot
    cond = handler['HostHaloID'] == -1
    halo_id = handler['ID'][cond]
    df_host = pd.DataFrame(halo_id, columns = ['ID'])
    for key in np.append(['Descendant', 'DescendantSnap'], mass_key):
        df_host[key] = handler[key][cond]
    df_host[mass_key] *= mass_unit
    df_host.set_index('ID', inplace = True)
    # Prepare progenitor data
    n_prog = len(prog_data)
    plain_data = []
    number = np.zeros(n_prog, dtype = int)
    for i_prog, prog in enumerate(prog_data.values()):
        number[i_prog] = len(prog)
        plain_data.extend(prog)
    df_prog = pd.DataFrame(plain_data, columns = ['ID', 'Snap'] + [k for k in mass_key])
    df_prog[mass_key] *= mass_unit
    df_prog.set_index('ID', inplace = True)
    # Set progenitor info
    df_host.insert(0, 'ProgIndex', -1)
    df_host.insert(1, 'ProgNumber', 0)
    df_host.loc[prog_data.keys(), 'ProgIndex'] = np.append(0, np.cumsum(number)[:-1])
    df_host.loc[prog_data.keys(), 'ProgNumber'] = number
    # Save data
    with pd.HDFStore(os.path.join(path, '%s_%03d.hdf5'%(prefix, snap)), "w") as store:
        store.append('Prog', df_prog)
        store.append('Host', df_host)


def SaveHaloConnector(handler, snap_min, snap_max, mass_key, path = './', prefix = 'hc'):
    mass_key = np.ravel(mass_key)
    n_snap = snap_max + 1
    desc_pool_arr = np.asarray([{} for snap in range(n_snap)])
    for snap in range(snap_min, n_snap):
        logger.info("Processing snaphost %d"%snap)
        handler.SwitchSnapshot(snap)
        # Merge subhalos to their host halo
        desc_pool = desc_pool_arr[snap]
        halo_id_arr = handler['ID']
        host_id_arr = handler['HostHaloID']
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
        SaveOutput(handler, desc_pool, mass_key, snap, path, prefix)
        # Walk forward
        desc_id_arr = handler['Descendant']
        desc_snap_arr = handler['DescendantSnap']
        props_arr = np.asarray([handler[k] for k in mass_key])
        for desc_id, desc_snap, host_id, halo_id, *props in zip(
            desc_id_arr, desc_snap_arr, host_id_arr, halo_id_arr, *props_arr
        ):
            if host_id == -1 and desc_snap <= snap_max:
                target_pool = desc_pool_arr[desc_snap]
                try:
                    target_pool[desc_id].append((halo_id, snap, *props))
                except KeyError:
                    target_pool[desc_id] = [(halo_id, snap, *props)]
        desc_pool_arr[snap] = None


def ReadHaloConnector(
    path, prefix, snap_min, snap_max, mass_key, redshifts, filters = {},
    progress_bar = False, kwargs_tqdm = {}
):
    # Select host halos
    fname = os.path.join(path, '%s_%03d.hdf5'%(prefix, snap_max))
    host_data = pd.read_hdf(
        fname, 'Host', mode = "r", columns = ['ProgIndex', 'ProgNumber', mass_key]
    )
    cond = np.full(len(host_data), True)
    for key, item in filters.items():
        cond &= item(host_data[key])
    n_host = np.sum(cond)
    # Create progress bar
    snap_range = range(snap_max, snap_min - 1, -1)
    if progress_bar:
        snap_range = tqdm(
            snap_range, desc = "Reading progenitors of %d halos"%n_host, **kwargs_tqdm
        )
    #
    for snap in snap_range:
        z = redshifts[snap - 1]
        # Read data
        with pd.HDFStore(os.path.join(path, '%s_%03d.hdf5'%(prefix, snap)), "r") as store:
            prog_data = pd.read_hdf(store, 'Prog', columns = ['Snap', mass_key]).to_records()
            if snap != snap_max:
                host_data = pd.read_hdf(
                    store, 'Host', columns = ['ProgIndex', 'ProgNumber', mass_key]
                )
            else:
                # Initialise root halos and the pool to read progenitors
                z_root = redshifts[snap_max]
                halo_id_arr = host_data.index[cond]
                m_root_arr = host_data.loc[halo_id_arr, mass_key]
                tree_arr = np.empty(n_host, dtype = object)
                pool = [[] for i_snap in range(snap_max + 1)]
                for i_tree, (halo_id, m_root) in enumerate(zip(halo_id_arr, m_root_arr)):
                    tree = HaloMergeTree(method = 'empty', n_base = 10**4)
                    tree.SetRedshifts(redshifts)
                    tree.Append(z_root, m_root)
                    tree_arr[i_tree] = tree
                    pool[snap_max].append((i_tree, 0, halo_id))
        # Trace progenitors
        current_pool = np.asarray(pool[snap])
        if len(current_pool) == 0:
            continue
        prog_pointer = host_data.loc[current_pool[:, -1], ['ProgIndex', 'ProgNumber']].values
        cond = prog_pointer[:, -1] > 0
        current_pool = current_pool[cond]
        prog_pointer = prog_pointer[cond]
        for (i_tree, idx_node_host, host_id), (idx, num) in zip(current_pool, prog_pointer):
            tree = tree_arr[i_tree]
            for i_prog, prog in enumerate(prog_data[idx:idx + num]):
                halo_id = prog['ID']
                halo_snap = prog['Snap']
                mass = prog[mass_key]
                idx_node_new = tree.Append(z, mass)
                if i_prog == 0:
                    # Append first progenitor
                    tree.LinkTailIndex(idx_node_host, idx_node_new)
                else:
                    # Append next progenitor
                    tree.LinkNextIndex(idx_node_prev, idx_node_new)
                pool[halo_snap].append((i_tree, idx_node_new, halo_id))
                idx_node_prev = idx_node_new
        pool[snap] = None
    return tree_arr


def ColumnLookup(store, columns):
    all_columns = []
    keys = list(store.keys())
    if '/Prog' in keys:
        keys.remove('/Prog')
    lookup = {k:[] for k in keys}
    for k in keys:
        for col in pd.read_hdf(store, k, start = 0, stop = 0).columns:
            all_columns.append(col)
            if col in columns:
                lookup[k].append(col)
    uni_columns, counts = np.unique(all_columns, return_counts = True)
    dup_columns = uni_columns[counts > 1]
    if len(dup_columns) > 0:
        raise ValueError("Duplicated columns: " + ", ".join(["'%s'"%name for name in dup_columns]))
    missing = [col for col in columns if col not in uni_columns]
    if len(missing) > 0:
        raise KeyError("Missing columns: " +  ", ".join(["'%s'"%name for name in missing]))
    keys.remove('/Host')
    for k in keys:
        if len(lookup[k]) == 0:
            del lookup[k]
    return lookup


def RemoveKey(hc_path, hc_prefix, snap_min, snap_max, key, progress_bar = True, kwargs_tqdm = {}):
    if key in ['Host', '/Host', 'Prog', '/Prog']:
        raise KeyError("'%s' is a protected key"%key)
    kwargs_tqdm.update(desc = "Removing '%s'"%key)
    for snap in SnapRange(snap_min, snap_max + 1, progress_bar = progress_bar, **kwargs_tqdm):
        with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap))) as store:
            try:
                store.remove(key)
            except KeyError:
                pass
