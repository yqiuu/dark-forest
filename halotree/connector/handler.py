import os
from copy import deepcopy
from abc import ABC, abstractmethod

from ..halotree import *

import numpy as np
import pandas as pd
import h5py
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from hmf.hmf import MassFunction

__all__ = [
    'SimHandler',
    'VELOCIraptorHandler',
]


class SimHandler(ABC):
    @abstractmethod
    def CacheSnapshot(self, snap):
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


class VELOCIraptorHandler(SimHandler):
    @property
    def little_h(self):
        try:
            return self._little_h
        except AttributeError:
            self._little_h = self.fp.get('Header')['Simulation'].attrs['h_val']
            return self._little_h


    @property
    def omega_m(self):
        try:
            return self._omega_m
        except AttributeError:
            self._omega_m = self.fp.get('Header')['Simulation'].attrs['Omega_m']
            return self._omega_m


    @property
    def cosmo_params(self):
        try:
            return dict(self._cosmo_params)
        except AttributeError:
            self._cosmo_params = {'H0':100*self.little_h, 'Om0':self.omega_m}
            return dict(self._cosmo_params)


    @property
    def volume(self):
        try:
            return self._volume
        except AttributeError:
            boxsize = self.fp.get('Header')['Simulation'].attrs['Period']
            self._volume = boxsize*boxsize*boxsize
            return self._volume


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
    def mass_conversion(self):
        """
        Returns
        -------
        float:
            A mass conversion factor to 10^10 h^-1 M_solar.
        """
        try:
            return self._mass_coversion
        except AttributeError:
            return 1e10*self.little_h


    @property
    def mass_unit(self):
        return 1e10*u.solMass


    @property
    def radius_unit(self):
        return u.megaparsec


    @property
    def velocity_unit(self):
        return u.km/u.second


    @property
    def m_particle(self):
        try:
            return self._m_particle
        except AttributeError:
            self.CacheSnapshot(0)
            self._m_particle = self['Mass_tot'][0]/self['npart'][0]*self.mass_conversion
            return self._m_particle


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


    @property
    def ages(self):
        """
        Returns
        -------
        array [Myr]
            Age of the universe at each time step
        """
        try:
            return np.array(self._ages)
        except AttributeError:
            cosmo = FlatLambdaCDM(**self.cosmo_params)
            self._ages = cosmo.age(self.redshifts).value*1e3
            return np.array(self._ages)


    def HaloKit(self, **kwargs):
        params = {'Mmin':5, 'dlog10m':0.05}
        params.update(kwargs)
        params['cosmo_params'] = self.cosmo_params
        # The following two parameters cannot be directly read from the output
        params['sigma_8'] = 0.815
        params['n'] = 0.9653
        hmf = MassFunction(**params)
        return HaloKit(hmf), hmf


    def HostID(self, n = 1):
        if n == 1:
            return -1
        else:
            return np.full(n, -1, dtype = self.data_type['hostHaloID'])


    def MaxForestID(self, snap_max):
        self.CacheSnapshot(0) # Assume that the snapshot starts at zero
        forest_id_max = max(self['ForestID'])
        for snap in range(1, snap_max + 1):
            self.CacheSnapshot(snap)
            forest_id_max = max(forest_id_max, *self['ForestID'])
        return forest_id_max


    def NewID(self, snap, inds):
        return snap*10**12 + inds


    def CacheSnapshot(self, snap):
        self.dataset = self.fp.get('Snap_%03d'%snap)


    def __getitem__(self, key):
        if key == 'DescendantSnap':
            return np.asarray(self.dataset['Descendant'])//10**12
        elif key == 'HostHaloID':
            return np.asarray(self.dataset['hostHaloID'])
        else:
            return np.asarray(self.dataset[key])


    def __enter__(self):
        self.fp = h5py.File(self.fname, "r")
        return self


    def __exit__(self, *args):
        self.fp.close()


    def __init__(self, fname):
        self.fname = fname
