from math import exp, log10

import numpy as np
from scipy import interpolate as intp
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf.hmf import MassFunction
from hmf._cache import cached_quantity


__all__ = ['EnhancedMassFunction']


class EnhancedMassFunction(MassFunction):
    @cached_quantity
    def alpha(self):
        lnm = np.log(self.m)
        lnsigma = spl(lnm, self.lnsigma)
        return lnsigma.derivative(n=1)(lnm)


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
