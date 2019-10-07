import os
from .utils import Range
from .halotree import *

import tables
import numpy as np
import pandas as pd

__all__ = [
    'SaveDarkForest',
    'ReadDarkForest'
]


def SaveSnapshot(hf, cntr, prog_data, mass_key, cols, snap):
        mass_conversion = cntr.mass_conversion
        # Save progenitor data
        group_prog = hf.create_group('/Snapshot_%03d'%snap, 'Prog', createparents=True)
        n_prog = len(prog_data)
        plain_data = []
        number = [0]*n_prog
        for i_prog, prog in enumerate(prog_data.values()):
            number[i_prog] = len(prog)
            plain_data.extend(prog)
        df_prog = pd.DataFrame(plain_data, columns=['Index', 'Snap'] + [k for k in mass_key])
        df_prog['Snap'] = df_prog['Snap'].astype('i4')
        df_prog[mass_key] *= mass_conversion
        if len(df_prog) > 0:
            for c in df_prog.columns:
                hf.create_carray(group_prog, c, obj=df_prog[c].values)
        # Save host data
        group_host = hf.create_group('/Snapshot_%03d'%snap, 'Host', createparents=True)
        select_host = cntr['HostHaloID'] == cntr.HostID()
        id_ = cntr['ID'][select_host]
        df_host = pd.DataFrame(id_, index=id_, columns=['ID'])
        df_host['ProgIndex'] = np.full(len(id_), -1, dtype='i8')
        df_host['ProgNumber'] = np.zeros(len(id_), dtype='i8')
        df_host.loc[prog_data.keys(), 'ProgIndex'] = np.append(0, np.cumsum(number)[:-1])
        df_host.loc[prog_data.keys(), 'ProgNumber'] = number
        hf.create_carray(group_host, 'Index', obj=id_)
        hf.create_carray(group_host, 'ProgIndex', obj=df_host['ProgIndex'].values)
        hf.create_carray(group_host, 'ProgNumber', obj=df_host['ProgNumber'].values)
        for k in mass_key:
            hf.create_carray(group_host, k, obj=cntr[k][select_host]*mass_conversion)
        for c in cols:
            hf.create_carray(group_host, c, obj=cntr[c][select_host])


def SaveHeader(hf, cntr):
    header = hf.create_group('/', 'Header')
    # Save redshifts
    hf.create_carray(header, 'Redshift', obj=cntr.redshifts)
    # Save attributes
    hf.set_node_attr(header, 'little_h', cntr.little_h)
    hf.set_node_attr(header, 'omega_m', cntr.omega_m)
    hf.set_node_attr(header, 'sigma_8', cntr.sigma_8)
    hf.set_node_attr(header, 'n_spectral', cntr.n_spectral)
    hf.set_node_attr(header, 'box_size', cntr.box_size)
    hf.set_node_attr(header, 'mass_unit_to_solarmass', cntr.mass_unit.value)
    hf.set_node_attr(header, 'length_unit_to_kpc', cntr.radius_unit.value)
    hf.set_node_attr(header, 'velocity_unit_to_kms', cntr.velocity_unit.value)
    hf.set_node_attr(header, 'm_particle', cntr.m_particle)


def SaveDarkForest(
    cntr, fname, mass_key, cols, snap_max, snap_min=0, filters=None, kwargs_tqdm={}
):
    kwargs_tqdm.update(desc="Saving dark forest")
    with tables.File('%s.hdf5'%fname, 'w', filters=filters) as hf:
        SaveHeader(hf, cntr)
        #
        mass_key = np.ravel(mass_key)
        cols = np.ravel(cols)
        n_snap = snap_max + 1
        desc_pool_arr = np.asarray([{} for snap in range(n_snap)])
        for snap in Range(snap_min, n_snap, **kwargs_tqdm):
            cntr.SwitchSnapshot(snap)
            # Merge subhalos to their host halo
            desc_pool = desc_pool_arr[snap]
            halo_id_arr = cntr['ID']
            host_id_arr = cntr['HostHaloID']
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
            SaveSnapshot(hf, cntr, desc_pool, mass_key, cols, snap)
            # Walk forward
            desc_id_arr = cntr['Descendant']
            desc_snap_arr = cntr['DescendantSnap']
            props_arr = np.asarray([cntr[k] for k in mass_key])
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


def ReadDarkForest(
    cntr, mass_key, snap_max, snap_min=0, m_range=None, kwargs_tqdm={}
):
    kwargs_tqdm.update(desc='Reading dark forest')
    #
    redshifts = cntr.redshifts
    # Select host halos
    cntr.SwitchSnapshot(snap_max)
    host_data = cntr.Read('Host', ['ProgIndex', 'ProgNumber', mass_key], 'pandas', 'Index')
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
            cntr.SwitchSnapshot(snap)
            host_data = cntr.Read('Host', ['ProgIndex', 'ProgNumber', mass_key], 'pandas', 'Index')
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
        prog_data = cntr.Read('Prog', ['Index', 'Snap', mass_key], 'records')
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
