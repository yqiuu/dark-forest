import os

from ..utils import SnapRange
from ..connector import ColumnLookup
from ..constructor import QuickConstructor

import h5py
import numpy as np
import pandas as pd


def SaveHeader(fp, handler, n_snap):
    grp = fp.create_group('Header')
    grp.attrs['NSnaps'] = n_snap
    #
    subgrp = grp.create_group('Simulation')
    subgrp.attrs['h_val'] = handler.little_h
    subgrp.attrs['Omega_m'] = handler.omega_m
    subgrp.attrs['Period'] = handler.volume**(1./3.)
    #
    subgrp = grp.create_group('Units')
    subgrp.attrs['Mass_unit_to_solarmass'] = handler.mass_conversion/handler.little_h


def SaveGroupData(
    fp, hc_path, hc_prefix, column_lookup, name_lookup, quick_cstr, data_type, snap, z
):
    grp = fp.create_group('Snap_%03d'%snap)
    with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap))) as store:
        for key, cols in column_lookup.items():
            df_save = pd.read_hdf(store, key, columns = cols)
            for c in df_save.columns:
                for c_save in name_lookup[c]:
                    grp.create_dataset(c_save, data = df_save[c].values, dtype = data_type[c_save])
    for key, cstr in quick_cstr.items():
        grp.create_dataset(key, data = cstr.Run(snap)[key].values, dtype = data_type[key])
    # Set attributes
    grp.attrs['Snapnum'] = snap
    grp.attrs['NHalos'] = len(df_save)
    grp.attrs['scalefactor'] = 1./(1. + z)


def VELOCIraptorConverter(
    handler, hc_path, hc_prefix, snap_min, snap_max, out_name,
    mass_key, other_key = [], progress_bar = True, kwargs_tqdm = {}
):
    data_type = handler.data_type
    mass_conversion = handler.mass_conversion
    #
    quick_cstr = {}
    for k_save, k_req in mass_key:
        quick_cstr[k_save] = QuickConstructor(
            handler, hc_path, hc_prefix, k_save, k_req, lambda m:m/mass_conversion
        )
    col_req = []
    name_lookup = {}
    for item in other_key:
        if isinstance(item, str):
            col_req.append(item)
            try:
                name_lookup[item].append(item)
            except KeyError:
                name_lookup[item] = [item]
        elif len(item) == 2:
            k_save, k_req = item
            col_req.append(k_req)
            try:
                name_lookup[k_req].append(k_save)
            except KeyError:
                name_lookup[k_req] = [k_save]
        elif len(item) == 3:
            k_save, k_req, transform = item
            quick_cstr[k_save] = QuickConstructor(
                handler, hc_path, hc_prefix, k_save, k_req, transform
            )
        else:
            raise ValueError('key ', item, 'is not understood')
    with pd.HDFStore(os.path.join(hc_path, '%s_%03d.hdf5'%(hc_prefix, snap_min))) as store:
        column_lookup = ColumnLookup(store, col_req)
    del column_lookup['/Host']
    #
    redshifts = handler.redshifts
    with h5py.File(out_name, "w") as fp:
        SaveHeader(fp, handler, snap_max - snap_min + 1)
        for snap in SnapRange(
            snap_min, snap_max + 1, progress_bar = progress_bar, **kwargs_tqdm
        ):
            z = redshifts[snap]
            SaveGroupData(
                fp, hc_path, hc_prefix, column_lookup, name_lookup, quick_cstr, data_type, snap, z
            )

