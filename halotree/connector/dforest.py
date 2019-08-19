from .connector import Connector

import tables
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


    def SwitchSnapshot(self, snap):
        self.group = self.hf.get_node('/Snapshot_%03d'%snap)


    def Load(self, target, cols=None):
        hf = self.hf
        group = hf.get_node(self.group, target)
        if cols is None:
            cols = [node.name for node in hf.iter_nodes(group)]
        data = {}
        for c in cols:
            data[c] = hf.get_node(group, c).read()
        return data


    def __getitem__(self, key):
        pass


    def __enter__(self):
        self.hf = tables.File(self.fname, 'r')
        return self


    def __exit__(self, *args):
        self.hf.close()


    def __init__(self, fname):
        self.fname = fname
