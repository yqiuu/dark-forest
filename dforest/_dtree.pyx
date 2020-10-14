from libc.math cimport sqrt
from libcpp.vector cimport vector

from .core import KitHalo

import numpy as np
from hmf.growth_factor import GrowthFactor


__all__ = [
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


cdef HaloRelation* halo_relation_init(kit_halo):
    cdef:
        double[::1] mv_m_arr = kit_halo.m_arr
        double[::1] mv_sigma_arr = kit_halo.sigma_arr
        double[::1] mv_alpha_arr = kit_halo.alpha_arr
        double[::1] mv_z_arr = kit_halo.z_arr
        double[::1] mv_delta_arr = kit_halo.delta_arr
        HaloRelation* hr = new HaloRelation(
            &mv_m_arr[0], &mv_sigma_arr[0],
            &mv_alpha_arr[0], mv_m_arr.shape[0],
            &mv_z_arr[0], &mv_delta_arr[0], mv_z_arr.shape[0]
        )
    return hr


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


    def trace_main_branch(self):
        cdef int idx_tail

        branch = []
        branch.append(self.data[0].mass)
        idx_tail = self.data[0].idx_tail
        while idx_tail != -1:
            branch.append(self.data[idx_tail].mass)
            idx_tail = self.data[idx_tail].idx_tail
        return np.array(branch)


    def trace_unevolved_subhalo(self):
        cdef int idx_tail
        cdef int idx_next

        branch = []
        branch.append(self.data[0].idx_next)
        idx_tail = self.data[0].idx_tail
        while idx_tail != -1:
            branch.append(self.data[idx_tail].idx_next)
            idx_tail = self.data[idx_tail].idx_tail
        m_sub_tot = []
        for idx_next0 in branch:
            m_sub_z = []
            idx_next = idx_next0
            while idx_next != -1:
                m_sub_z.append(self.data[idx_next].mass)
                idx_next = self.data[idx_next].idx_next
            m_sub_tot.append(np.asarray(m_sub_z))
        return np.asarray(m_sub_tot)


cdef class TreeBinarySplit:
    """Wrapper class to generator merger tree by binary splits."""
    cdef HaloRelation* hr 
    cdef gsl_rng* rstate
    cdef BinarySplit* grower


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
        self.hr = halo_relation_init(
            core if isinstance(core, KitHalo) else core.kit_halo()
        )
        self.rstate = gsl_rng_init(seed)
        self.grower = new BinarySplit(
            self.hr, self.rstate, m_res, G0, gamma1, gamma2, eps1, eps2
        )


    def __dealloc__(self):
        gsl_rng_free(self.rstate)
        del self.hr
        del self.grower


    def grow(self, m_root, z_target):
        cdef double[::] mv = np.asarray(z_target, dtype=np.double).flatten()
        cdef vector[double] vec
        vec.assign(&mv[0], &mv[0] + mv.shape[0])
        tree = MergerTree()
        tree.from_vector(self.grower.grow(m_root, vec))
        return tree


cdef class TreeEnsemble:
    cdef HaloRelation* hr 
    cdef gsl_rng* rstate
    cdef EnsembleGrower* grower


    def __cinit__(self,
        object core,
        double m_res=1e8,
        double eps=0.1,
        int n_draw=100,
        unsigned long int seed=0
    ):
        self.hr = halo_relation_init(
            core if isinstance(core, KitHalo) else core.kit_halo()
        )
        self.rstate = gsl_rng_init(seed)
        self.grower = new EnsembleGrower(
            self.hr, self.rstate, m_res, eps, n_draw
        )


    def __dealloc__(self):
        gsl_rng_free(self.rstate)
        del self.hr
        del self.grower


    def grow(self, m_root, z_target):
        cdef double[::] mv = np.asarray(z_target, dtype=np.double).flatten()
        cdef vector[double] vec
        vec.assign(&mv[0], &mv[0] + mv.shape[0])
        tree = MergerTree()
        tree.from_vector(self.grower.grow_tree(m_root, vec))
        return tree


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
