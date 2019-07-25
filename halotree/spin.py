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
        handler = self.handler
        handler.CacheSnapshot(np.argmin(abs(self.redshifs - z)))
        spin_sample = np.log10(handler['lambda_B'][handler['HostHaloID'] == handler.HostID()])
        spin = 10**gaussian_kde(spin_sample).resample(n_halo).flatten()
        return spin


    def __init__(self, handler):
        self.handler = handler
        self.redshifs = handler.Redshifts()
