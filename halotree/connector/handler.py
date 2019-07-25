import os
from abc import ABC, abstractmethod

from ..halotree import *

import numpy as np
import pandas as pd
import h5py
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
    def _GetDataType(self):
        data_type = {}
        with h5py.File(self.fname, "r") as fp:
            dataset = fp.get('Snap_000')
            for key in list(dataset.keys()):
                data_type[key] = dataset[key].dtype
        self.data_type = data_type


    def HubbleConstant(self):
        return self.fp.get('Header')['Simulation'].attrs['h_val']


    def OmegaM(self):
        return self.fp.get('Header')['Simulation'].attrs['Omega_m']


    def Volume(self):
        boxsize = self.fp.get('Header')['Simulation'].attrs['Period']
        return boxsize*boxsize*boxsize


    def HaloKit(self, **kwargs):
        params = {'Mmin':5, 'dlog10m':0.05}
        params.update(kwargs)
        params['cosmo_params'] = {'H0':100*self.HubbleConstant(), 'Om0':self.OmegaM()}
        # The following two parameters cannot be directly read from the output
        params['sigma_8'] = 0.815
        params['n'] = 0.9653
        hmf = MassFunction(**params)
        return HaloKit(hmf), hmf


    def MassConversion(self):
        """
        Returns
        -------
        float:
            A mass conversion factor to 10^10 h^-1 M_solar.
        """
        return 1e10*self.HubbleConstant()


    def NSnap(self):
        n_snap = 0
        keys = list(self.fp.keys())
        for k in keys:
            if k.startswith('Snap'):
                n_snap += 1
        return n_snap


    def Redshifts(self, snap_max = -1):
        if snap_max == -1:
            n_snap = self.NSnap()
        else:
            n_snap = snap_max + 1
        redshifts = np.zeros(n_snap)
        for snap in range(n_snap):
            redshifts[snap] = 1./self.fp.get('Snap_%03d'%snap).attrs['scalefactor'] - 1.
        return redshifts


    def Ages(self, snap_max = -1):
        """
        Returns
        -------
        array [Myr]
            Age of the universe at each time step
        """
        cosmo = FlatLambdaCDM(H0 = 100*self.HubbleConstant(), Om0 = self.OmegaM())
        return cosmo.age(self.Redshifts(snap_max)).value*1e3


    def HostID(self, n = 1):
        if n == 1:
            return -1
        else:
            return np.full(n, -1, dtype = self.data_type['hostHaloID'])


    def MaxForestID(self):
        self.CacheSnapshot(0)
        forest_id_max = max(self['ForestID'])
        for snap in range(1, self.NSnap()):
            self.CacheSnapshot(snap)
            forest_id_max = max(forest_id_max, *self['ForestID'])
        return forest_id_max


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
        self._GetDataType()
