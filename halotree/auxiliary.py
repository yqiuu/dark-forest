import numpy as np


def EstimateConditionalMassFunction(progenitors, m_root, mass_bins = None):
    if mass_bins is None:
        mass_bins = np.logspace(8, 12, 10)
    mass_mid = mass_bins[:-1] + .5*np.diff(mass_bins)
    dlogm = np.diff(np.log(mass_bins))
    func = np.zeros([len(progenitors), len(mass_mid)])
    for i_tree in range(len(progenitors)):
        halos = progenitors[i_tree]['mass']
        for i_bin, (l, u) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
            func[i_tree, i_bin] = np.sum(halos[(halos > l) & (halos < u)])/m_root
    return mass_mid, func.mean(axis = 0)/dlogm
        
