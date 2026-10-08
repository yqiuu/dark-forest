"""Read VELOCIraptor catalogues and export reconstructed halo trees."""

import gc

from ..logger import logger, dividing_line, memory_usage
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
    """Read a VELOCIraptor HDF5 tree catalogue as a context manager."""

    def __init__(self, fname):
        self.fname = fname


    def __enter__(self):
        self.fp = h5py.File(self.fname, "r")
        self.hf = tables.File(self.fname, "r")
        return self


    def __exit__(self, *args):
        self.fp.close()
        self.hf.close()


    def __getitem__(self, key):
        """Read ``(snapshot, column)``, translating host and descendant fields."""
        snap, column = key
        dataset = self.fp.get('Snap_%03d'%snap)
        if column == 'DescendantSnap':
            return np.asarray(dataset['Descendant'])//10**12
        elif column == 'HostHaloID':
            return np.asarray(dataset['hostHaloID'])
        else:
            return np.asarray(dataset[column])


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
        """Infer particle mass in h^-1 solar masses from the first halo at snapshot 50."""
        try:
            return self._m_particle
        except AttributeError:
            self._m_particle = self[50, 'Mass_tot'][0]/self[50, 'npart'][0]*self.mass_conversion
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


    def max_forest_id(self, snap_max, snap_min):
        """Find the largest forest ID over an inclusive range starting with halos."""
        forest_id_max = max(self[snap_min, 'ForestID']) # Assume that the snapshot starts at zero
        for snap in range(snap_min, snap_max + 1):
            forest_id_max = max(forest_id_max, *self[snap, 'ForestID'])
        return forest_id_max


    def host_id(self, n=1):
        """Return the host-halo sentinel (-1), as a scalar or an array of length n."""
        if n == 1:
            return -1
        else:
            return np.full(n, -1, dtype = self.data_type['hostHaloID'])


    def new_id(self, snap, inds):
        """Encode zero-based halo indices and a snapshot number as catalogue IDs."""
        return snap*10**12 + inds + 1


class VELOCIraptorSaver:
    """Export source and reconstructed properties to VELOCIraptor HDF5.

    Property items are names or tuples of (output name, replacement name,
    optional transform). ``items_mass`` are converted to source mass units.
    """

    def __init__(self,
        cntr_ext, cntr_src, save_name, items_mass, items_norm,
        snap_max, snap_min=0, kwargs_tqdm={}
    ):
        self.cntr_ext = cntr_ext
        self.cntr_src = cntr_src
        self.save_name = save_name
        self.args_mass = self._set_args(items_mass, True)
        self.args_norm = self._set_args(items_norm, False)
        self.snap_max = snap_max
        self.snap_min = snap_min
        kwargs_tqdm.update(desc="Save properties")
        self.kwargs_tqdm = kwargs_tqdm
        self.data_type = cntr_src.data_type


    def run(self):
        """Write the combined catalogue for the inclusive snapshot range."""
        scale_factors = 1./(1. + self.cntr_ext.redshifts)
        with tables.File(self.save_name, 'w') as hf_out:
            self._save_header(hf_out, self.cntr_src.hf, self.snap_max - self.snap_min + 1)
            for snap in trange(self.snap_min, self.snap_max + 1, **self.kwargs_tqdm):
                logger.info(f"Processing snapshot {snap}")
                grp = hf_out.create_group('/', 'Snap_%03d'%snap)
                # Save Reversed properties
                self._save_reversed(hf_out, grp, snap, scale_factors[snap])
                # Save halo masses
                self._save_cols(hf_out, grp, snap, self.args_mass)
                # Save other halo properties
                self._save_cols(hf_out, grp, snap, self.args_norm)
                logger.info(dividing_line())


    def _report(self, message):
        logger.info(f"{message} ({memory_usage()})")


    def _save_header(self, hf_out, hf_src, n_snap):
        header = hf_out.create_group('/', 'Header')
        hf_out.set_node_attr(header, 'NSnaps', n_snap)
        for grp_src in hf_src.iter_nodes('/Header'):
            grp_out = hf_out.create_group(header, grp_src._v_name)
            for name in grp_src._v_attrs._f_list():
                hf_out.set_node_attr(grp_out, name, hf_src.get_node_attr(grp_src, name))


    def _save_cols(self, hf_out, grp, snap, list_args):
        for args in list_args:
            c_target = args['c_target']
            arr = np.ravel(self.cntr_ext.gather(self.cntr_src, snap, **args))
            arr = arr.astype(self.data_type[c_target])
            self._report(f"Saving {c_target}")
            hf_out.create_carray(grp, c_target, obj=arr)
            gc.collect()
        return len(arr)


    def _save_reversed(self, hf_out, grp, snap, a):
        # Save Index and LeafType
        leaf_type = self.cntr_ext.read_cols(snap, 'Host', ['Index', 'LeafType'])
        data = self.cntr_ext.gather(self.cntr_src, snap, 'ID')
        data['LeafType'] = 0
        data.loc[leaf_type['Index'], 'LeafType'] = leaf_type['LeafType']
        self._report("Saving reversed properties")
        hf_out.create_carray(grp, 'Index', obj=data.index.values.astype('i8'))
        hf_out.create_carray(grp, 'LeafType', obj=data['LeafType'].values.astype('i4'))
        # Save attributes
        hf_out.set_node_attr(grp, 'NHalos', len(data))
        hf_out.set_node_attr(grp, 'Snapnum', snap)
        hf_out.set_node_attr(grp, 'scalefactor', a)
 

    def _set_args(self, items, is_mass):
        list_args = [None]*len(items)
        for i_it, it in enumerate(items):
            args = {}
            it = np.ravel(it)
            args['c_target'] = it[0]
            if len(it) > 1:
                args['c_rep'] = it[1]
            if len(it) > 2:
                args['transform'] = it[2]
            args['is_mass'] = is_mass
            list_args[i_it] = args
        return list_args
