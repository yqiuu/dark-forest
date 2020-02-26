import os
from .utils import Range
from .halotree import *

import tables
import numpy as np
import pandas as pd

__all__ = [
    'ReadDarkForest'
]


def ReadDarkForest(
    cntr, mass_key, snap_max, snap_min=0, m_range=None, kwargs_tqdm={}
):
    kwargs_tqdm.update(desc='Reading dark forest')
    #
    redshifts = cntr.redshifts
    # Select host halos
    host_data = cntr.Read(
        snap_max, 'Host', ['ProgIndex', 'ProgNumber', mass_key], 'pandas', 'Index'
    )
    if m_range is not None:
        m_min, m_max = m_range
        m_host = host_data[mass_key].values
        cond = (m_host >= m_min) & (m_host < m_max)
    else:
        cond = np.full(len(host_data), True)
    n_host = np.count_nonzero(cond)
    #
    for snap in Range(snap_max, snap_min, -1, **kwargs_tqdm):
        z = redshifts[snap - 1]
        # Read data
        if snap != snap_max:
            host_data = cntr.Read(
                snap, 'Host', ['ProgIndex', 'ProgNumber', mass_key], 'pandas', 'Index'
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
        prog_data = cntr.Read(snap, 'Prog', ['Index', 'Snap', mass_key], 'records')
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
                halo_id = prog['Index']
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
