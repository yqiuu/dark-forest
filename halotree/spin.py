from abc import ABC, abstractmethod


from .logger import CreateLogger

import numpy as np
from scipy.stats import gaussian_kde


logger = CreateLogger(__name__)


class SpinSampler(ABC):
    @abstractmethod
    def Draw(self, mass, z):
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass


class KernelSpinSampler(SpinSampler):
    def Draw(self, n_halo, z):
        # Load local variables
        handler = self.handler
        redshifs = self.redshifs
        mass_key = self.mass_key
        m_cut = self.m_cut
        n_min = self.n_min
        snap_fit = self.snap_fit
        #
        snap_target = max(snap_fit, np.argmin(abs(self.redshifs - z)))
        while True:
            handler.CacheSnapshot(snap_target)
            spin_sample = handler['lambda_B'][
                (handler['HostHaloID'] == handler.HostID()) & (handler[mass_key] > m_cut)
            ]
            if len(spin_sample) > n_min:
                break
            else:
                snap_target += 1
                self.snap_fit = snap_target
        spin = 10**gaussian_kde(np.log10(spin_sample)).resample(n_halo).flatten()
        return spin


    def __init__(self, handler, mass_key, m_cut = 0., n_min = 100000):
        self.handler = handler
        self.redshifs = handler.redshifts
        self.mass_key = mass_key
        self.m_cut = m_cut/handler.mass_conversion
        self.n_min = n_min
        self.snap_fit = 0 # Assume that the snapshot starts at zero
