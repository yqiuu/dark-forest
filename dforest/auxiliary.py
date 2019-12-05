import numpy as np


def EstimateConditionalMassFunction(m_prog, m_root, mass_bins = None):
    if mass_bins is None:
        mass_bins = np.logspace(8, 12, 10)
    lnmass_bins = np.log(mass_bins)
    dlnmass = np.diff(lnmass_bins)
    mass_mid = np.exp((lnmass_bins[1:] + 0.5*dlnmass))
    func = np.zeros([len(m_prog), len(mass_mid)])
    for i_tree in range(len(m_prog)):
        halos = m_prog[i_tree]
        for i_bin, (l, u) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
            func[i_tree, i_bin] = np.sum(halos[(halos > l) & (halos < u)])/m_root
    return mass_mid, func.mean(axis = 0)/dlnmass
