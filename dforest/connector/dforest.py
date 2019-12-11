import os

from .connector import Connector

import tables
import numpy as np
import pandas as pd
from tables.exceptions import NoSuchNodeError
from tqdm import trange
from astropy import units as u


__all__ = ['DarkForestConnector']


class DarkForestConnector(Connector):
    @property
    def little_h(self):
        try:
            return self._little_h
        except AttributeError:
            self._little_h = self.hf.get_node_attr('/Header', 'little_h')
            return self._little_h


    @property
    def omega_m(self):
        try:
            return self._omega_m
        except AttributeError:
            self._omega_m = self.hf.get_node_attr('/Header', 'omega_m')
            return self._omega_m


    @property
    def sigma_8(self):
        try:
            return self._sigma_8
        except AttributeError:
            self._sigma_8 = self.hf.get_node_attr('/Header', 'sigma_8')
            return self._sigma_8


    @property
    def n_spectral(self):
        try:
            return self._n_spectral
        except AttributeError:
            self._n_spectral = self.hf.get_node_attr('/Header', 'n_spectral')
            return self._n_spectral


    @property
    def box_size(self):
        try:
            return self._box_size
        except AttributeError:
            self._box_size = self.hf.get_node_attr('/Header', 'box_size')
            return self._box_size

    
    @property
    def mass_unit(self):
        try:
            return self._mass_unit
        except AttributeError:
            self._mass_unit = self.hf.get_node_attr('/Header', 'mass_unit_to_solarmass')*u.solMass
            return self._mass_unit


    @property
    def radius_unit(self):
        try:
            return self._radius_unit
        except AttributeError:
            self._radius_unit = self.hf.get_node_attr('/Header', 'length_unit_to_kpc')*u.kiloparsec
            return self._radius_unit


    @property
    def velocity_unit(self):
        try:
            return self._velocity_unit
        except AttributeError:
            self._velocity_unit = \
                self.hf.get_node_attr('/Header', 'velocity_unit_to_kms')*u.km/u.second
            return self._velocity_unit


    @property
    def m_particle(self):
        try:
            return self._m_particle
        except AttributeError:
            self._m_particle = self.hf.get_node_attr('/Header', 'm_particle')
            return self._m_particle


    @property
    def redshifts(self):
        try:
            return np.array(self._redshifts)
        except AttributeError:
            self._redshifts = self.hf.get_node('/Header', 'Redshift').read()
            return np.array(self._redshifts)


    def SwitchSnapshot(self, snap):
        groups = []
        for f in self.files:
            groups.append(f.get_node('/Snapshot_%03d'%snap))
        self.groups = groups


    def Read(self, target, cols=None, rtype='dict', index=None):
        """
        Read data.

        Parameters
        ----------
            target: {'Host', 'Prog', 'Main'}
                Name of the group.
            cols: list
                List of string specifying column names.
            rtype: {'dict', 'pandas', 'records'}
                Type of the return data.
                    dict: python dictionary
                    pandas: pandas data frame
                    records: numpy records
            index: str
                If rtype is 'pandas', specify the index of the data frame.

        Returns
        -------
            Required data.
        """
        n_file = len(self.files)
        data_all = [None]*n_file
        offset = [0]*(n_file + 1)
        n_tot = 0
        for i_f, (hf, grp) in enumerate(zip(self.files, self.groups)):
            grp = hf.get_node(grp, target)
            if i_f == 0:
                if cols is None:
                    cols = [node.name for node in hf.iter_nodes(grp)]
                else:
                    cols = list(cols)
                if index is not None:
                    if index not in cols:
                        cols.append(index)
            data = {}
            for c in cols:
                data[c] = hf.get_node(grp, c).read()
            data_all[i_f] = data
            n_tot += len(data[c])
            offset[i_f + 1] = n_tot

        data_out = {}
        inds_offset = 0
        for c in cols:
            arr = np.zeros(n_tot, dtype=data_all[0][c].dtype)
            for i_f in range(n_file):
                a = data_all[i_f][c]
                if c == 'Index':
                    cond = a < 0
                    a[cond] -= inds_offset
                    inds_offset += np.count_nonzero(cond)
                arr[offset[i_f] : offset[i_f + 1]] = a
            data_out[c] = arr
        # Change data type
        if rtype == 'dict':
            return data_out
        data_out = pd.DataFrame.from_dict(data_out)
        if rtype == 'pandas':
            if index is not None:
                data_out.set_index(index, inplace=True)
            return data_out
        elif rtype == 'records':
            return data_out.to_records(index=False)
        else:
            raise ValueError(f"Unkown rtype: '{rtype}'")


    def Remove(self, name, kwargs_tqdm={}):
        kwargs_tqdm.update(desc="Removing '%s'"%name)
        n_snap = len(self.hf.list_nodes('/')) - 1
        for i_snap in trange(n_snap, **kwargs_tqdm):
            for f in self.files:
                try:
                    f.remove_node('/Snapshot_%03d'%i_snap, name, recursive=True)
                except NoSuchNodeError:
                    pass


    def __getitem__(self, key):
        pass


    def __enter__(self):
        if self.write:
            mode = 'r+'
            filters = self.filters
        else:
            mode = 'r'
            filters = None
        files = []
        for f in self.fname:
            files.append(tables.File(f, mode, filters=filters))
        self.files = files
        self.hf = files[0]
        return self


    def __exit__(self, *args):
        for f in self.files:
            f.close()


    def __init__(self, fname, prefix=None, write=False, filters=None):
        if prefix is None:
            self.fname = [fname]
            self.write = write
            self.filters = filters
        else:
            files = []
            for f in os.listdir(fname):
                if f.startswith(prefix):
                    files.append(os.path.join(fname, f))
            self.fname = files
            self.write = False
            self.filters = None
