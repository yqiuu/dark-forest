from ..halotree import *
from .connector import Connector

import numpy as np
import pandas as pd
import h5py
import tables
from tqdm import trange
from astropy import units as u


__all__ = [
    'VELOCIraptorConnector',
    'VELOCIraptorSaver',
]


class VELOCIraptorConnector(Connector):
    @property
    def little_h(self):
        try:
            return self._little_h
        except AttributeError:
            self._little_h = self.fp['Header/Simulation'].attrs['h_val']
            return self._little_h


    @property
    def omega_m(self):
        try:
            return self._omega_m
        except AttributeError:
            self._omega_m = self.fp['Header/Simulation'].attrs['Omega_m']
            return self._omega_m


    @property
    def sigma_8(self):
        return 0.815


    @property
    def n_spectral(self):
        return 0.9653


    @property
    def box_size(self):
        """Box size. [Mpc]"""
        try:
            return self._box_size
        except AttributeError:
            box_size = self.fp['Header/Simulation'].attrs['Period']/self.little_h
            box_size = (box_size*self.radius_unit).to(u.megaparsec).value
            self._box_size = box_size
            return box_size


    @property
    def mass_unit(self):
        try:
            return self._mass_unit
        except AttributeError:
            self._mass_unit = self.fp['Header/Units'].attrs['Mass_unit_to_solarmass']*u.solMass
            return self._mass_unit


    @property
    def radius_unit(self):
        try:
            return self._radius_unit
        except AttributeError:
            self._radius_unit = self.fp['Header/Units'].attrs['Length_unit_to_kpc']*u.kiloparsec
            return self._radius_unit


    @property
    def velocity_unit(self):
        try:
            return self._velocity_unit
        except AttributeError:
            self._velocity_unit = \
                self.fp['Header/Units'].attrs['Velocity_unit_to_kms']*u.km/u.second
            return self._velocity_unit


    @property
    def m_particle(self):
        try:
            return self._m_particle
        except AttributeError:
            self._m_particle = self[0, 'Mass_tot'][0]/self[0, 'npart'][0]*self.mass_conversion
            return self._m_particle


    @property
    def data_type(self):
        try:
            return dict(self._data_type)
        except AttributeError:
            data_type = {}
            with h5py.File(self.fname, "r") as fp:
                dataset = fp.get('Snap_000')
                for key in list(dataset.keys()):
                    data_type[key] = dataset[key].dtype
            self._data_type = data_type
            return dict(self._data_type)


    @property
    def n_snap(self):
        try:
            return self._n_snap
        except AttributeError:
            n_snap = 0
            keys = list(self.fp.keys())
            for k in keys:
                if k.startswith('Snap'):
                    n_snap += 1
            self._n_snap = n_snap
            return n_snap


    @property
    def redshifts(self):
        try:
            return np.array(self._redshifts)
        except AttributeError:
            n_snap = self.n_snap
            redshifts = np.zeros(n_snap)
            for snap in range(n_snap):
                redshifts[snap] = 1./self.fp.get('Snap_%03d'%snap).attrs['scalefactor'] - 1.
            self._redshifts = redshifts
            return np.array(self._redshifts)


    def HostID(self, n = 1):
        if n == 1:
            return -1
        else:
            return np.full(n, -1, dtype = self.data_type['hostHaloID'])


    def MaxForestID(self, snap_max):
        forest_id_max = max(self[0, 'ForestID']) # Assume that the snapshot starts at zero
        for snap in range(1, snap_max + 1):
            forest_id_max = max(forest_id_max, *self[snap, 'ForestID'])
        return forest_id_max


    def NewID(self, snap, inds):
        return snap*10**12 + inds + 1


    def __getitem__(self, key):
        snap, column = key
        dataset = self.fp.get('Snap_%03d'%snap)
        if column == 'DescendantSnap':
            return np.asarray(dataset['Descendant'])//10**12
        elif column == 'HostHaloID':
            return np.asarray(dataset['hostHaloID'])
        else:
            return np.asarray(dataset[column])


    def __enter__(self):
        self.fp = h5py.File(self.fname, "r")
        self.hf = tables.File(self.fname, "r")
        return self


    def __exit__(self, *args):
        self.fp.close()
        self.hf.close()

    def __init__(self, fname):
        self.fname = fname


class VELOCIraptorSaver:
    def _SaveHeader(self, hf_out, hf_src, n_snap):
        header = hf_out.create_group('/', 'Header')
        hf_out.set_node_attr(header, 'NSnaps', n_snap)
        for grp_src in hf_src.iter_nodes('/Header'):
            grp_out = hf_out.create_group(header, grp_src._v_name)
            for name in grp_src._v_attrs._f_list():
                hf_out.set_node_attr(grp_out, name, hf_src.get_node_attr(grp_src, name))


    def _SaveColumns(self, hf_out, grp, items, snap):
        cols = [None]*len(items)
        items_algin = [None]*len(items)
        for i_it, it in enumerate(items):
            it = tuple(np.ravel(it))
            if len(it) == 1:
                items_algin[i_it] = (it[0], it[0], None)
            elif len(it) == 2:
                items_algin[i_it] = (*it, None)
            elif len(it) == 3:
                items_algin[i_it] = it
            else:
                raise ValueError(f"Unknown items {c}")
            cols[i_it] = it[0]
        data = self.cntr_ext.Gather(snap, self.cntr_src, cols, self.cols_link, self.cols_mass)
        for name_c, name_s, transform in items_algin:
            if callable(transform):
                arr = transform(data[name_c].values).astype(self.data_type[name_s])
            else:
                arr = data[name_c].values.astype(self.data_type[name_s])
            hf_out.create_carray(grp, name_s, obj=arr)
        return len(data)


    def _SaveReversed(self, hf_out, grp, snap, a):
        # Save Index and LeafType
        leaf_type = self.cntr_ext.Read(snap, 'Host', ['Index', 'LeafType'])
        data = self.cntr_ext.Gather(snap, self.cntr_src, [])
        data['LeafType'] = 0
        data.loc[leaf_type['Index'], 'LeafType'] = leaf_type['LeafType']
        hf_out.create_carray(grp, 'Index', obj=data.index.values.astype('i8'))
        hf_out.create_carray(grp, 'LeafType', obj=data['LeafType'].values.astype('i4'))
        # Save attributes
        hf_out.set_node_attr(grp, 'NHalos', len(data))
        hf_out.set_node_attr(grp, 'Snapnum', snap)
        hf_out.set_node_attr(grp, 'scalefactor', a)
 

    def __init__(self,
        cntr_ext, cntr_src, save_name, items_link, items_mass, items_norm,
        snap_max, snap_min=0, kwargs_tqdm={}
    ):
        self.cntr_ext = cntr_ext
        self.cntr_src = cntr_src
        self.save_name = save_name
        self.items_link = items_link
        self.items_mass = items_mass
        self.items_norm = items_norm
        self.cols_link = [np.ravel(item)[0] for item in items_link]
        self.cols_mass = [np.ravel(item)[0] for item in items_mass]
        self.snap_max = snap_max
        self.snap_min = snap_min
        self.kwargs_tqdm = kwargs_tqdm
        #
        data_type = cntr_src.data_type
        data_type['Index'] = 'i8'
        data_type['LeafType'] = 'i4'
        self.data_type = data_type


    def Run(self):
        scale_factors = 1./(1. + self.cntr_ext.redshifts)
        with tables.File(self.save_name, 'w') as hf_out:
            self._SaveHeader(hf_out, self.cntr_src.hf, self.snap_max - self.snap_min + 1)
            for snap in trange(self.snap_min, self.snap_max + 1, **self.kwargs_tqdm):
                grp = hf_out.create_group('/', 'Snap_%03d'%snap)
                # Save Reversed properties
                self._SaveReversed(hf_out, grp, snap, scale_factors[snap])
                # Save Link properties
                self._SaveColumns(hf_out, grp, self.items_link, snap)
                # Save halo masses
                self._SaveColumns(hf_out, grp, self.items_mass, snap)
                # Save other halo properties
                self._SaveColumns(hf_out, grp, self.items_norm, snap)
