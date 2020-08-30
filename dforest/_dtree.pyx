from libc.math cimport sqrt
from libcpp.vector cimport vector

import struct as binary

import numpy as np
from hmf.growth_factor import GrowthFactor


__all__ = [
    'HaloKit',
    'ProgFunc',
    'TreeBinarySplit',
    'TreeEnsemble',
    'estimate_conditional_mass_func'
]


cdef struct c_Leaf:
    double mass
    int level
    int idx_tail
    int idx_next


cdef gsl_rng* gsl_rng_init(unsigned long int seed):
    cdef gsl_rng* rstate = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rstate, seed)
    return rstate


cdef class MergerTree:
    """Container of a single tree. Data are stored in a c++ vector."""
    cdef vector[Leaf] data


    cdef void from_vector(self, vector[Leaf]&& d):
        self.data = d


    def __init__(self, n_base=kInitialSize):
        self.data.reserve(n_base)


    cpdef int append(self, mass, level):
        self.data.push_back(Leaf(mass, level, -1, -1))
        return self.data.size() - 1


    cpdef void link_tail_index(self, idx, idx_tail):
        self.data[idx].idx_tail = idx_tail


    cpdef void link_next_index(self, idx, idx_next):
        self.data[idx].idx_next = idx_next


    def leaf(self, level=-1):
        data = np.asarray(<c_Leaf[:self.data.size()]>(<c_Leaf*>&self.data[0]))
        m_halo = np.copy(data['mass'])
        if level == -1:
            return m_halo
        else:
            return m_halo[data['level'] == level]


    def tree(self):
        return np.copy(<c_Leaf[:self.data.size()]>(<c_Leaf*>&self.data[0]))


cdef class TreeBinarySplit:
    """Wrapper class to generator merger tree by binary splits."""
    cdef HaloKit halo_kit
    cdef gsl_rng* rstate
    cdef BinarySplit* planter


    def __cinit__(self,
        object core,
        double G0=0.57,
        double gamma1=0.38,
        double gamma2=-0.01,
        double m_res=1e8,
        double eps1=.1,
        double eps2=.1,
        unsigned long int seed=0
    ):
        self.halo_kit = core if isinstance(core, HaloKit) else core.halo_kit()
        self.rstate = gsl_rng_init(seed)
        self.planter = new BinarySplit(
            self.halo_kit.hr, self.rstate, m_res,
            G0, gamma1, gamma2, eps1, eps2
        )


    def __dealloc__(self):
        gsl_rng_free(self.rstate)
        del self.planter


    def grow(self, m_root, z_root, z_target):
        cdef double[::] mv = np.asarray(z_target, dtype=np.double).flatten()
        cdef vector[double] vec
        vec.assign(&mv[0], &mv[0] + mv.shape[0])
        tree = MergerTree()
        tree.from_vector(self.planter.grow(m_root, z_root, vec))
        return tree


cdef class TreeEnsemble:
    cdef gsl_rng* rstate
    cdef EnsembleGrower* grower


    def __cinit__(self,
        ProgFunc pmf,
        double m_res=1e8,
        int n_draw=100,
        unsigned long int seed=0
    ):
        self.rstate = gsl_rng_init(seed)
        self.grower = new EnsembleGrower(
            pmf.prog_func, self.rstate, m_res, n_draw
        )


    def __dealloc__(self):
        gsl_rng_free(self.rstate)
        del self.grower


    def grow(self, m_root, z_root, z_target):
        cdef double[::] mv = np.asarray(z_target, dtype=np.double).flatten()
        cdef vector[double] vec
        vec.assign(&mv[0], &mv[0] + mv.shape[0])
        tree = MergerTree()
        tree.from_vector(self.grower.grow_tree(m_root, z_root, vec))
        return tree


cdef class ProgFunc:
    cdef ProgMassFunction* prog_func


    def __cinit__(self, HaloKit halo_kit):
        self.prog_func = new ProgMassFunction(halo_kit.hr)


    def __dealloc__(self):
        del self.prog_func


def rebuild_halo_kit(*rebuild_args):
    return HaloKit(None, rebuild_args = rebuild_args)


cdef class HaloKit:
    cdef:
        HaloRelation* hr
        tuple rebuild_args
    cdef:
        public double m_min
        public double m_max


    def __cinit__(self, hmf, z_grid=None, rebuild_args=None):
        if rebuild_args is None:
            hmf.update(z = 0)
            self.m_min = 10**hmf.Mmin
            self.m_max = 10**hmf.Mmax
            # Compute sigma_m
            m_arr = np.copy(hmf.m)
            sigma_arr = np.copy(hmf.sigma)
            # Compute alpha_m
            alpha_arr = np.copy(-hmf._dlnsdlnm)
            # Compute delta_z
            if z_grid is None:
                z_arr = np.linspace(0, 200, 1000)
            else:
                z_arr = np.copy(z_grid)
            gf = GrowthFactor(hmf.cosmo)
            delta_arr \
                = hmf.delta_c/np.array([gf.growth_factor(z) for z in z_arr])
        else:
            self.m_min, self.m_max, \
            m_arr, sigma_arr, alpha_arr, z_arr, delta_arr = rebuild_args
        # Set rebuild args
        self.rebuild_args = (
            self.m_min, self.m_max,
            m_arr, sigma_arr, alpha_arr, z_arr, delta_arr
        )
        # Initialisation
        cdef:
            double[::1] mv_m_arr = m_arr
            double[::1] mv_sigma_arr = sigma_arr
            double[::1] mv_alpha_arr = alpha_arr
            double[::1] mv_z_arr = z_arr
            double[::1] mv_delta_arr = delta_arr
        self.hr = new HaloRelation(
            &mv_m_arr[0], &mv_sigma_arr[0],
            &mv_alpha_arr[0], mv_m_arr.shape[0],
            &mv_z_arr[0], &mv_delta_arr[0], mv_z_arr.shape[0]
        )


    def __dealloc__(self):
        del self.hr


    def __reduce__(self):
        return (rebuild_halo_kit, self.rebuild_args)


    def save_binary(self, fname):
        """Save a binary file of necessary arrays for external uses."""
        self.m_min, self.m_max, m_arr, \
        sigma_arr, alpha_arr, z_arr, delta_arr = self.rebuild_args
        with open(fname, "wb") as fp:
            n_mass = m_arr.size
            fp.write(binary.pack('i', n_mass))
            fp.write(binary.pack('%dd'%n_mass, *m_arr))
            fp.write(binary.pack('%dd'%n_mass, *sigma_arr))
            fp.write(binary.pack('%dd'%n_mass, *alpha_arr))

            n_z = z_arr.size
            fp.write(binary.pack('i', n_z))
            fp.write(binary.pack('%dd'%n_z, *z_arr))
            fp.write(binary.pack('%dd'%n_z, *delta_arr))


def estimate_conditional_mass_func(tree, level, frac_bins=None, mean=True):
    tree = np.ravel(tree)
    level = np.ravel(level)
    # Set fraction bins
    if frac_bins is None:
        frac_bins = np.logspace(-4, 0, 21)
    lnfrac_bins = np.log(frac_bins)
    dlnfrac = np.diff(lnfrac_bins)
    frac_mid = np.exp(lnfrac_bins[:-1] + .5*dlnfrac)
    #
    cmf = np.zeros([len(tree), len(level), len(frac_mid)])
    for i_tree, tree_single in enumerate(tree):
        m_root = tree_single.m_root
        mass_bins = frac_bins*m_root
        for i_level, level_single in enumerate(level):
            m_prog = tree_single.halos(level_single)['mass']
            for i_bin, (l, u) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
                cmf[i_tree, i_level, i_bin] = np.sum(m_prog[(m_prog > l) & (m_prog < u)])/m_root
    #
    if mean:
        cmf = cmf.mean(axis = 0)
    return frac_mid, cmf/dlnfrac
