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
            spin_sample = handler[snap_target, 'lambda_B'][
                (handler[snap_target, 'HostHaloID'] == handler.HostID()) \
                & (handler[snap_target, mass_key] > m_cut)
            ]
            if len(spin_sample) > n_min:
                break
            else:
                snap_target += 1
                self.snap_fit = snap_target
        cond = spin_sample != 0.
        if len(spin_sample) != np.count_nonzero(cond):
            logger.warning("Find halos with zero spin.")
        return np.ravel(10**gaussian_kde(np.log10(spin_sample[cond])).resample(n_halo))


    def __init__(self, handler, mass_key, m_cut = 0., n_min = 100000):
        self.handler = handler
        self.redshifs = handler.redshifts
        self.mass_key = mass_key
        self.m_cut = m_cut/handler.mass_conversion
        self.n_min = n_min
        self.snap_fit = 0 # Assume that the snapshot starts at zero
