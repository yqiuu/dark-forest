from math import sqrt, exp, log10
import struct as binary

import numpy as np
from numpy import vectorize
from scipy import interpolate as intp
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf._cache import cached_quantity
from hmf.growth_factor import GrowthFactor
from hmf.hmf import MassFunction


__all__ = [
    'sample_mass_func',
    'smart_bins',
    'EnhancedMassFunction',
    'KitHalo'
]


def sample_mass_func(dndm, m_hmf, m_min, m_max, volume, rstate=None):
    """
    Draw samples from the halo mass function in a given volume.

    Parameters
    ----------
    m_min: float [h^-1 M_solar]
        Minimum mass of the resulting sample.
    m_max: float [h^-1 M_solar]
        Maximum mass of the resulting sample.
    volume: float [h^-3 Mpc]
        Effective volume for the resulting sample.

    Returns
    -------
    1-D array
         A sample of masses which follows the given halo mass function.
    """
    if rstate is None:
        rstate = np.random.RandomState()
    hmf_spl = spl(m_hmf, dndm)
    m_cdf = np.append(m_min, m_hmf[(m_hmf > m_min) & (m_hmf < m_max)])
    m_cdf = np.append(m_cdf, m_max)
    cdf = np.asarray(list(map(lambda m: hmf_spl.integral(m_min, m), m_cdf)))
    cdf, inds =  np.unique(cdf, return_index=True)
    m_cdf = m_cdf[inds]
    n_sample = int(cdf[-1]*volume)
    cdf /= cdf[-1]
    cdf_spl = spl(cdf, m_cdf)
    return cdf_spl(rstate.rand(n_sample))


def smart_bins(data, n_min, d_min, scale='linear'):
    """
    Generate bin edges given the minimum number of points in each bin
    and the minimum bin width.

    Parameters
    ----------
        data: array_like
            Input data.
        n_min:
            Minimum number of points in each bin.
        d_min:
            Minimum bin width.
        scale: {'linear', 'log'}, optional.
            Bin scale.

    Returns
    -------
        array_like
            Bin edges.
    """
    if scale == 'linear':
        op1 = lambda a, b: a - b
    elif scale == 'log':
        d_min = 10**d_min
        op1 = lambda a, b: a/b
    else:
        raise ValueError("Unknown scale type '%s'"%scale)

    data = np.sort(data)
    b_min = data[0]
    b_curr = data[-1]
    bins = [b_curr]
    while len(data) > 0:
        if len(data) <= n_min:
            bins.append(b_min)
            break
        else:
            b_next = data[-n_min]
        if op1(b_curr, b_next) > d_min:
            bins.append(b_next)
        else:
            b_next = max(b_min, op1(b_curr, d_min))
            bins.append(b_next)
        data = data[:data.searchsorted(b_next)]
        b_curr = b_next
    bins.reverse()
    return np.asarray(bins)


class EnhancedMassFunction(MassFunction):
    @cached_quantity
    def bias(self):
        """
        Halo bias using the fitting formula of Tinker et al. 2010.
        """
        nu = np.sqrt(self.nu)
        delta_c = self.delta_c
        y = log10(self.delta_h)
        exp_factor = exp(-(4./y)**4)
        A = 1. + .24*y*exp_factor
        a = .44*y - 0.88
        B = .183
        b = 1.5
        C = .019 + .107*y + .19*exp_factor
        c = 2.4
        return 1 - A*nu**a/(nu**a + delta_c**a) + B*nu**b + C*nu**c


    def prog_mass_func(self, m_prog, z_prog, m_root, z_root, frac=True):
        #
        delta_c = self.delta_c
        m = self.m
        sigma = self.sigma
        alpha = -self._dlnsdlnm
        #
        spl_sigmasq = spl(m, sigma*sigma)
        sigmasq_m = spl_sigmasq(m_prog)
        sigmasq_root = spl_sigmasq(m_root)
        diff_sigmasq = sigmasq_m - sigmasq_root
        alpha_m = spl(m, alpha)(m_prog)
        #
        gf = GrowthFactor(self.cosmo)
        diff_delta = delta_c/gf.growth_factor(z_prog) - delta_c/gf.growth_factor(z_root)
        #
        f_exp = np.exp(-.5*diff_delta*diff_delta/diff_sigmasq)
        pmf = sqrt(2./np.pi)*diff_delta*sigmasq_m*diff_sigmasq**-1.5*f_exp*alpha_m
        if frac:
            return pmf
        else:
            return m_root/m_prog*pmf


    def conditional_mass_func(self, delta_lin, m_cell, m_halo, z):
        z0 = self.z
        cosmo = self.cosmo
        n_eff = self.n_eff
        mass = self.m
        mean_density0 = self.mean_density0
        delta_c = self.delta_c
        delta_halo = self.delta_halo
        hmf_model = self.hmf_model
        hmf_params = self.hmf_params
        self.update(z=0.)
        sigma = self.sigma
        alpha = -self._dlnsdlnm
        self.update(z=z0)
        #
        spl_sigma = spl(mass, sigma, ext=1)
        sigma_halo = spl_sigma(m_halo)
        sigma_cell = spl_sigma(m_cell)
        gf = GrowthFactor(cosmo).growth_factor(z)
        spl_alpha = spl(mass, alpha, ext=1)
        alpha_halo = spl_alpha(m_halo)
        #
        diff_delta = (delta_c - delta_lin)/gf
        diff_sigma2 = np.ravel(sigma_halo*sigma_halo - sigma_cell*sigma_cell)
        cond = diff_sigma2 > 0
        diff_sigma2[~cond] = 1.
        nu2 = diff_delta*diff_delta/diff_sigma2
        nu2 = nu2[cond]
        fsigma = np.zeros(len(diff_sigma2))
        fsigma[cond] = hmf_model(
            nu2=nu2, z=z, delta_halo=delta_halo, omegam_z=cosmo.Om(z),
            delta_c=delta_c, n_eff=n_eff, **hmf_params
        ).fsigma
        return mean_density0*alpha_halo/(m_halo*m_halo)*sigma_halo*sigma_halo/diff_sigma2*fsigma


    def halo_density(self, delta, v_cell, m_halo, z):
        mean_density0 = self.mean_density0
        m_max = 10**(self.Mmax + 1)
        ratio = 1 + delta
        delta_lin = -1.35*ratio**(-2./3.) + .78785*ratio**-0.58661 - 1.12431*ratio**-.5 + 1.68647
        m_cell = mean_density0*v_cell*ratio
        den_halo = self.conditional_mass_func(delta_lin, m_cell, m_halo, z) \
            / self.conditional_mass_func(0., m_max, m_halo, z) * ratio
        return den_halo


    def sample_mass_func(self, m_min, m_max, volume, rstate=None):
        return sample_mass_func(self.dndm, self.m, m_min, m_max, volume, rstate)


class KitHalo:
    def __init__(self, hmf, z_grid=None):
        hmf.update(z=0)
        # Compute sigma_m
        self.m_arr = np.copy(hmf.m)
        self.sigma_arr = np.copy(hmf.sigma)
        # Compute alpha_m
        self.alpha_arr = np.copy(-hmf._dlnsdlnm)
        # Compute delta_z
        if z_grid is None:
            self.z_arr = np.linspace(0, 200, 1000)
        else:
            self.z_arr = np.copy(z_grid)
        gf = GrowthFactor(hmf.cosmo)
        self.delta_arr \
            = hmf.delta_c/np.array([gf.growth_factor(z) for z in self.z_arr])
        #
        self.m_to_sigma = spl(self.m_arr, self.sigma_arr)
        self.sigma_to_m = spl(self.sigma_arr[::-1], self.m_arr[::-1])
        self.alpha = spl(self.m_arr, self.alpha_arr)
        self.z_to_d = spl(self.z_arr, self.delta_arr)
        self.d_to_z = spl(self.delta_arr, self.z_arr)
        #
        self.norm_cmf = -np.sqrt(2/np.pi)*np.log(10)


    def cmf(self, m1, z1, m2, z2, frac=True):
        @vectorize
        def limit(ds):
            if ds == 0.:
                return 0.
            else:
                return ds**-1.5*np.exp(-.5*dd*dd/ds)

        dd = self.z_to_d(z1) - self.z_to_d(z2)
        sigma1 = self.m_to_sigma(m1)
        sigmasq1 = sigma1*sigma1
        sigma2 = self.m_to_sigma(m2)
        sigmasq2 = sigma2*sigma2
        ds = sigmasq1 - sigmasq2
        alpha = self.alpha(m1)
        cmf = self.norm_cmf*sigmasq1*dd*limit(ds)*alpha
        if frac:
            return cmf
        else:
            return cmf*m2/m1


    def save_binary(self, fname):
        """Save a binary file of necessary arrays for external uses."""
        with open(fname, "wb") as fp:
            n_mass = self.m_arr.size
            fp.write(binary.pack('i', n_mass))
            fp.write(binary.pack('%dd'%n_mass, *self.m_arr))
            fp.write(binary.pack('%dd'%n_mass, *self.sigma_arr))
            fp.write(binary.pack('%dd'%n_mass, *self.alpha_arr))

            n_z = self.z_arr.size
            fp.write(binary.pack('i', n_z))
            fp.write(binary.pack('%dd'%n_z, *self.z_arr))
            fp.write(binary.pack('%dd'%n_z, *self.delta_arr))

