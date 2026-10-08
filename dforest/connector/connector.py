"""Common cosmology and unit interface for halo catalogue connectors."""

import os
from abc import ABC, abstractmethod
from copy import deepcopy

from ..core import EnhancedMassFunction
from ..logger import logger
from .._dtree import *

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G


__all__ = ['Connector']


class Connector(ABC):
    """Expose snapshot columns, cosmology, and units through a context manager.

    Subclasses implement column access as ``connector[snap, column]``.
    """

    @abstractmethod
    def __init__(self, fp):
        pass


    @abstractmethod
    def __enter__(self):
        pass


    @abstractmethod
    def __exit__(self, *args):
        pass


    @abstractmethod
    def __getitem__(self, key):
        pass


    @property
    @abstractmethod
    def little_h(self):
        return 0.


    @property
    @abstractmethod
    def omega_m(self):
        return 0.


    @property
    @abstractmethod
    def sigma_8(self):
        return 0.


    @property
    @abstractmethod
    def n_spectral(self):
        return 0.


    @property
    def cosmo_params(self):
        try:
            return dict(self._cosmo_params)
        except AttributeError:
            self._cosmo_params = {'H0': 100*self.little_h, 'Om0': self.omega_m}
            return dict(self._cosmo_params)


    @property
    @abstractmethod
    def box_size(self):
        """Box size. [Mpc]"""
        return 0.


    @property
    def volume(self):
        """Box volume. [Mpc^3]"""
        try:
            return self._volume
        except AttributeError:
            self._volume = self.box_size**3
            return self._volume


    @property
    def mass_conversion(self):
        """A multiplier to convert input mass unit to 10^10 h^-1 M_solar."""
        try:
            return self._mass_coversion
        except AttributeError:
            self._mass_coversion = (self.mass_unit*self.little_h).value
            return self._mass_coversion


    @property
    @abstractmethod
    def mass_unit(self):
        """An astropy unit (quantity) instance specifying the mass unit."""
        return 1e10*u.solMass


    @property
    @abstractmethod
    def radius_unit(self):
        """An astropy unit (quantity) instance specifying the length unit."""
        return u.megaparsec


    @property
    @abstractmethod
    def velocity_unit(self):
        """An astropy unit (quantity) instance specifying the velocity unit."""
        return u.km/u.second


    @property
    @abstractmethod
    def m_particle(self):
        """Particle mass. [h^-1 M_solar]"""
        return 0.


    @property
    @abstractmethod
    def redshifts(self):
        """Redshift of the universe at each snapshot."""
        return np.array([0.])


    @property
    def ages(self):
        """Age of the universe at each snapshot. [Myr]"""
        try:
            return np.array(self._ages)
        except AttributeError:
            cosmo = FlatLambdaCDM(**self.cosmo_params)
            self._ages = cosmo.age(self.redshifts).value*1e3
            return np.array(self._ages)


    @property
    def G(self):
        """Gravitational constant.

        The unit is determined by the given length mass and velocity units.
        """
        return G.to(self.radius_unit/self.mass_unit*self.velocity_unit**2).value


    def update_emf_params(self, z_grid=None, **kwargs):
        """Update mass-function parameters and the HaloKit redshift grid."""
        self._z_grid = z_grid
        try:
            params = self._params
        except AttributeError:
            params = {
                'Mmin': 5,
                'dlog10m': 0.05,
                'cosmo_params': self.cosmo_params,
                'sigma_8': self.sigma_8,
                'n': self.n_spectral
            }
            self._params = params
        params.update(**kwargs)


    def emf(self):
        """Create an enhanced mass function using this connector's cosmology."""
        self.update_emf_params()
        return EnhancedMassFunction(**self._params)


    def halo_kit(self):
        """Create the interpolation tables needed to grow Monte Carlo trees."""
        return HaloKit(self.emf(), self._z_grid)


    def rho_halo(self, z, mode='crit', delta_h=200.):
        """Return halo density in catalogue mass/length^3 units.

        ``mode`` selects virial, critical, or mean density; ``delta_h`` sets
        the overdensity for the latter two choices.
        """
        density_unit = self.mass_unit/self.radius_unit**3
        cosmo = FlatLambdaCDM(**self.cosmo_params)
        rho_crit = cosmo.critical_density(z).to(density_unit).value
        if mode == 'vir':
            x = self.cosmo.Om(z) - 1
            rho_halo = (18.*np.pi*np.pi + 82.*x - 39.*x*x)*rho_crit
        elif mode == 'crit':
            rho_halo = delta_h*rho_crit
        elif mode == 'mean':
            rho_halo = delta_h*cosmo.Om(z)*rho_crit
        else:
            raise ValueError("Choose mode from 'vir', 'crit' and 'mean'")
        return rho_halo


    def _init_emf(self):
        self._emf = EnhancedMassFunction(
            Mmin=5,
            dlog10m=0.05,
            cosmo_params=self.cosmo_params,
            sigma_8=self.sigma_8,
            n=self.n_spectral
        )
        return self._emf
