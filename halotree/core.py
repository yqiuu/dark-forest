from math import sqrt, exp, log10

import numpy as np
from numpy import vectorize
from scipy import interpolate as intp
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf._cache import cached_quantity
from hmf.growth_factor import GrowthFactor
from hmf.hmf import MassFunction


__all__ = ['EnhancedMassFunction']


class EnhancedMassFunction(MassFunction):
    def ProgenitorMassFunction(self, m_prog, z_prog, m_root, z_root, frac=True):
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


    def ConditionalMassFunction(self, delta_lin, m_cell, m_halo, z):
        z0 = self.z
        self.update(z=0.)
        mass = self.m
        sigma = self.sigma
        alpha = -self._dlnsdlnm
        mean_density0 = self.mean_density0
        delta_c = self.delta_c
        self.update(z=z0)
        #
        spl_sigma = spl(mass, sigma, ext=1)
        sigma_halo = spl_sigma(m_halo)
        sigma_cell = spl_sigma(m_cell)
        gf = GrowthFactor(self.cosmo).growth_factor(z)
        spl_alpha = spl(mass, alpha, ext=1)
        alpha_halo = spl_alpha(m_halo)
        #
        delta_lin = np.ravel(delta_lin)
        f_sigma = np.ravel(sigma_halo*sigma_halo - sigma_cell*sigma_cell)
        f1 = np.zeros(len(f_sigma), dtype=f_sigma.dtype)
        cond = f_sigma > 0.
        if len(delta_lin) == 1:
            f_delta = (delta_c - delta_lin)/gf
        else:
            f_delta = (delta_c - delta_lin[cond])/gf
        f_sigma = f_sigma[cond]
        f1[cond] = f_delta*f_sigma**-1.5*np.exp(-.5*f_delta*f_delta/f_sigma)
        f2 = sqrt(2/np.pi)*sigma_halo*sigma_halo*alpha_halo*mean_density0/(m_halo*m_halo)
        return f1*f2


    def HaloDensity(self, delta, v_cell, m_halo, z):
        mean_density0 = self.mean_density0
        m_max = 10**(self.Mmax + 1)

        delta_lin = -1.35*(1 + delta)**(-2./3.) + .78785*(1 + delta)**-0.58661 \
            - 1.12431*(1 + delta)**-.5 + 1.68647
        m_cell = (1 + delta)*mean_density0*v_cell
        den_halo = self.ConditionalMassFunction(delta_lin, m_cell, m_halo, z) \
            / self.ConditionalMassFunction(0., m_max, m_halo, z) * (1 + delta)
        return den_halo


    def Bias(self, m=None):
        """
        Compute the halo bias using the fitting formula of Tinker et al. 2010.
        """
        nu = self.nu
        delta_c = self.delta_c
        y = log10(self.delta_h)
        exp_factor = exp(-(4./y)**4)
        A = 1. + .24*y*exp_factor
        a = .44*y - 0.88
        B = .183
        b = 1.5
        C = .019 + .107*y + .19*exp_factor
        c = 2.4
        bias = 1 - A*nu**a/(nu**a + delta_c**a) + B*nu**b + C*nu**c
        if m is None:
            return bias
        else:
            return spl(self.m, bias)(m)
