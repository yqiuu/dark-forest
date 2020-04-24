from libc.math cimport sqrt
from libcpp.vector cimport vector
from _dtree cimport HaloRelation as cpp_HaloRelation, \
    KitSplit, Leaf, BinarySplit, gsl_rng, \
    gsl_rng_type, gsl_rng_alloc, gsl_rng_free, gsl_rng_mt19937

import struct as binary
from .logger import logger

import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf.growth_factor import GrowthFactor


__all__ = [
    'RandomStateGSL',
    'GrowKit',
    'HaloKit',
    'HaloMergeTree',
    'tree_split',
    'estimate_conditional_mass_func'
]


cdef struct c_Leaf:
    double mass
    int level
    int idx_tail
    int idx_next


cdef class RandomStateGSL:
    """
    Wrapper of the GSL random number generator.
    """

    cdef gsl_rng* rstate

    def __cinit__(self, seed):
        self.rstate = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self.rstate, seed)


    def __init__(self, seed):
        pass


    def __dealloc__(self):
        gsl_rng_free(self.rstate)


cdef class GrowKit:
    cdef KitSplit* kit_sp
    cdef public double m_res


    def __cinit__(self, G0, gamma1, gamma2, m_res, eps1, eps2):
        self.kit_sp = new KitSplit(G0, gamma1, gamma2, m_res, eps1, eps2)
        self.m_res = m_res

    def __dealloc__(self):
        del self.kit_sp


def rebuild_halo_kit(*rebuild_args):
    return HaloKit(None, rebuild_args = rebuild_args)


cdef class HaloKit:
    cdef:
        HaloRelation* hr
        tuple rebuild_args
    cdef:
        public double m_min
        public double m_max


    def __cinit__(self, hmf, z_grid=None, save=None, rebuild_args=None):
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
            delta_arr = hmf.delta_c \
                /np.array([gf.growth_factor(z) for z in z_arr])
        else:
            self.m_min, self.m_max, m_arr, \
            sigma_arr, alpha_arr, z_arr, delta_arr = rebuild_args
        # Set rebuild args
        self.rebuild_args = (
            self.m_min, self.m_max, m_arr,
            sigma_arr, alpha_arr, z_arr, delta_arr
        )
        # Save a binary file of necessary arrays for external uses
        if save is not None:
            with open(save, "wb") as fp:
                n_mass = m_arr.size
                fp.write(binary.pack('i', n_mass))
                fp.write(binary.pack('%dd'%n_mass, *m_arr))
                fp.write(binary.pack('%dd'%n_mass, *sigma_arr))
                fp.write(binary.pack('%dd'%n_mass, *alpha_arr))

                n_z = z_arr.size
                fp.write(binary.pack('i', n_z))
                fp.write(binary.pack('%dd'%n_z, *z_arr))
                fp.write(binary.pack('%dd'%n_z, *delta_arr))
        # Initialisation
        cdef:
            double[::1] mv_m_arr = m_arr
            double[::1] mv_sigma_arr = sigma_arr
            double[::1] mv_alpha_arr = alpha_arr
            double[::1] mv_z_arr = z_arr
            double[::1] mv_delta_arr = delta_arr
        self.hr = new HaloRelation(
            &mv_m_arr[0], &mv_sigma_arr[0], &mv_alpha_arr[0], mv_m_arr.shape[0],
            &mv_z_arr[0], &mv_delta_arr[0], mv_z_arr.shape[0]
        )


    def __dealloc__(self):
        del self.hr


    def __reduce__(self):
        return (rebuild_halo_kit, self.rebuild_args)


cdef class HaloMergeTree:
    cdef:
        vector[Leaf] tree
        object z_arr


    def __cinit__(self, method='split', **params):
        self.z_arr = None
        if method == 'split':
            self._init_split(
                params['halo_kit'], params['grow_kit'], params['m_root'],
                params['z_root'], params['z_target'], params['m_res'],
                params['rstate'], params['n_base']
            )
        elif method == 'empty':
            self._init_empty(params['n_base'])
        else:
            raise ValueError("Method %s is not understood"%method)


    property m_root:
        def __get__(self):
            return self.tree[0].mass


    property redshifts:
        def __get__(self):
            return np.copy(self.z_arr)
        def __set__(self, z_arr):
            if self.z_arr is None:
                self.z_arr = np.sort(z_arr)
            else:
                logger.warning("redshifts can only be set once")


    def levels(self, z, tol=0.01):
        level = np.argmin(abs(self.redshifts - z))
        z_level = self.redshifts[level]
        if abs(z_level - z) > tol:
            logger.warning("Fail to find a redshift within the tolerance")
        return level


    def halos(self, level=-1):
        # Create a copy of all nodes with modified attribute name
        tree_arr = np.asarray(<c_Leaf[:self.tree.size()]>(<c_Leaf*>&self.tree[0]))
        if level != -1:
            tree_arr = tree_arr[tree_arr['level'] == level]
        #print(tree_arr)
        dtype = [
            ('mass', 'f8'), ('z', 'f8'), ('idx_tail', 'i4'), ('idx_next', 'i4')
        ]
        node_arr = np.zeros(len(tree_arr), dtype=dtype)
        node_arr['mass'] = np.copy(tree_arr['mass'])
        node_arr['z'] = self.redshifts[tree_arr['level']]
        node_arr['idx_tail'] = np.copy(tree_arr['idx_tail'])
        node_arr['idx_next'] = np.copy(tree_arr['idx_next'])
        return node_arr


    cdef _check_mass_range(self, HaloKit halo_kit, m_root, m_res):
        if m_root <= halo_kit.m_min or m_root >= halo_kit.m_max:
            raise ValueError("m_root beyond mass interpolation range")
        if m_res <= halo_kit.m_min or m_res >= halo_kit.m_max:
            raise ValueError("m_res beyond mass interpolation range")


    cpdef int append(self, z, mass):
        self.tree.push_back(Leaf(mass, <int>z, -1, -1))


    cpdef void link_tail_index(self, idx, idx_tail):
        self.tree[idx].idx_tail = idx_tail


    cpdef void link_next_index(self, idx, idx_next):
        self.tree[idx].idx_next = idx_next


#    cdef _convert_redshifts(self, double z_root, double[::1] z_target):
#        z_arr = np.append(z_root, z_target)
#        self.redshifts = z_arr
#        node_arr = np.asarray(<Node[:self.tree.n_node]>self.tree.data)
#        for i_delta, delta in enumerate(np.unique(node_arr['trace'])):
#            trace = node_arr['trace']
#            trace[trace == delta] = z_arr[i_delta]


    cdef _init_empty(self, int n_base):
        self.tree.reserve(n_base)


    cdef _init_split(self,
        HaloKit halo_kit, GrowKit grow_kit, double m_root, double z_root,
        double[::1] z_target, double m_res, RandomStateGSL rstate, int n_base
    ):
        self._check_mass_range(halo_kit, m_root, m_res)
        # Create the tree
        cdef vector[double] z_target_vec
        z_target_vec.assign(&z_target[0], &z_target[-1] + 1)
        self.tree = BinarySplit(
            halo_kit.hr, grow_kit.kit_sp, rstate.rstate
        ).grow(m_root, z_root, z_target_vec)
        z_arr = np.append(z_root, z_target)
        self.redshifts = z_arr
        # Convert delta to redshift
        #self._convert_redshifts(z_root, z_target)


def tree_split(halo_kit, grow_kit, m_root, z_root, z_target, rstate=None, n_base=100000):
    z_target = np.ravel(z_target).astype('f8')
    if rstate is None:
        rstate = RandomStateGSL(209204)
    return HaloMergeTree(
        method='split', halo_kit=halo_kit, grow_kit=grow_kit, m_root=m_root, z_root=z_root,
        z_target=z_target, m_res=grow_kit.m_res, rstate=rstate, n_base=n_base
    )


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
