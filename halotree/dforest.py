import os
from .utils import SnapRange
from .halotree import *

import tables
import numpy as np
import pandas as pd

__all__ = [
    'SaveDarkForest',
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
        df_prog = pd.DataFrame(plain_data, columns=['ID', 'Snap'] + [k for k in mass_key])
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
            hf.create_carray(group_host, c, obj=cntr[c])


def SaveHeader(hf, cntr):
    header = hf.create_group('/', 'Header')
    # Save redshifts
    hf.create_carray(header, 'Redshift', obj=cntr.redshifts)
    # Save attributes
    hf.set_node_attr(header, 'little_h', cntr.little_h)
    hf.set_node_attr(header, 'omega_m', cntr.omega_m)
    hf.set_node_attr(header, 'n_spectral', cntr.n_spectral)
    hf.set_node_attr(header, 'box_size', cntr.box_size)
    hf.set_node_attr(header, 'mass_unit_to_solarmass', cntr.mass_unit.value)
    hf.set_node_attr(header, 'length_unit_to_kpc', cntr.radius_unit.value)
    hf.set_node_attr(header, 'velocity_unit_to_kms', cntr.velocity_unit.value)


def SaveDarkForest(
    cntr, fname, mass_key, cols, snap_max, snap_min=0, filters=None, kwargs_tqdm={}
):
    with tables.File('%s.hdf5'%fname, 'w', filters=filters) as hf:
        SaveHeader(hf, cntr)
        #
        mass_key = np.ravel(mass_key)
        cols = np.ravel(cols)
        n_snap = snap_max + 1
        desc_pool_arr = np.asarray([{} for snap in range(n_snap)])
        for snap in SnapRange(snap_min, n_snap, **kwargs_tqdm):
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
