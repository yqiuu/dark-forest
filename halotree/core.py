from math import sqrt, exp, log10

import numpy as np
from scipy import interpolate as intp
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf._cache import cached_quantity
from hmf.growth_factor import GrowthFactor
from hmf.hmf import MassFunction


__all__ = ['EnhancedMassFunction']


class EnhancedMassFunction(MassFunction):
    @cached_quantity
    def alpha(self):
        lnm = np.log(self.m)
        lnsigma = spl(lnm, self.lnsigma)
        return lnsigma.derivative(n=1)(lnm)


    def ConditionalMassFunction(self, m_prog, z_prog, m_root, z_root):
        #
        delta_c = self.delta_c
        m = self.m
        sigma = self.sigma
        alpha = -self.alpha
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
        factor_exp = np.exp(-.5*diff_delta*diff_delta/diff_sigmasq)
        return sqrt(2./np.pi)*diff_delta*sigmasq_m*diff_sigmasq**-1.5*factor_exp*alpha_m


    def ProgenitorMassFunction(self, m_prog, z_prog, m_root, z_root):
        return m_root/m_prog*self.ConditionalMassFunction(m_prog, z_prog, m_root, z_root)


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
