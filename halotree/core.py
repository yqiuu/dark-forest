from math import exp, log10

from scipy import interpolate as intp
from hmf.hmf import MassFunction


__all__ = ['EnhancedMassFunction']


class EnhancedMassFunction(MassFunction):
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
            return intp.splev(m, intp.splrep(self.m, bias, s=0))
