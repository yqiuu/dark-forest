import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as Spline


def SampleHaloMassFunction(m_min, m_max, m_hmf, dndm, volume, rstate = None):
    """
    Draw samples according to a given halo mass function in a given volume.
    Mass units should be consistent and in linear scale. Volume units should be
    consistent.
    
    Parameters
    ----------
        m_min: float
            Minimum mass of the resulting sample.
        m_max: float
            Maximum mass of the resulting sample.
        m_hmf: numpy array
            X-axis of the given halo mass function.
        dndm: numpy array
            Number density of halos per unit of mass.
        volume: float
            Effective volume for the resulting sample.

    Returns
    -------
        numpy array
             A sample of masses which follows the given halo mass function.
    """
    if rstate is None:
        rstate = np.random.RandomState()
    hmf_spl = Spline(m_hmf, dndm)
    m_cdf = np.append(m_min, m_hmf[(m_hmf > m_min) & (m_hmf < m_max)])
    m_cdf = np.append(m_cdf, m_max)
    cdf = np.asarray(list(map(lambda m:hmf_spl.integral(m_min, m), m_cdf)))
    cdf, inds =  np.unique(cdf, return_index = True)
    m_cdf = m_cdf[inds]
    n_sample = int(cdf[-1]*volume)
    cdf /= cdf[-1]
    cdf_spl = Spline(cdf, m_cdf)
    return cdf_spl(rstate.rand(n_sample))
