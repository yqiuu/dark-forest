from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

import struct as binary

from .logger import CreateLogger

import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf.growth_factor import GrowthFactor


logger = CreateLogger(__name__)


cdef extern from "<gsl/gsl_rng.h>":
    ctypedef struct gsl_rng:
        pass

    ctypedef struct gsl_rng_type:
        pass

    gsl_rng_type* gsl_rng_mt19937

    gsl_rng* gsl_rng_alloc(gsl_rng_type *T)
    void gsl_rng_set(gsl_rng *r, unsigned long int s)
    void gsl_rng_free(gsl_rng *r)


cdef extern from "clib/halotree.h":
    ctypedef struct HaloRelation:
        pass

    HaloRelation HaloRelation_Read(char* fanme)
    HaloRelation HaloRelation_Init(
        double *mass, double *sigma, double *alpha, int n_mass, double *z, double *delta_c, int n_z
    )
    void HaloRelation_Free(HaloRelation *hr)


    ctypedef struct SplitKit:
        double G0
        double gamma1
        double gamma2
        double m_res
        double eps1
        double eps2

    double SplitKit_J(SplitKit *kit_sp, double u_res)
    SplitKit SplitKit_Init(
        double G0, double gamma1, double gamma2, double m_res, double eps1, double eps2
    )
    void SplitKit_Free(SplitKit *kit_sp)


    ctypedef struct Node:
        double trace
        double mass
        int idx_tail
        int idx_next

    ctypedef struct NodeVector:
        int n_max
        int n_node
        Node* data

    int NodeVector_Append(NodeVector* vector, double delta, double mass)
    NodeVector NodeVector_Init(int n_base)
    void NodeVector_Free(NodeVector *vector)

    NodeVector MergeTree_Grow(
        double m_root, double z_root, double *z_target, int n_target,
        HaloRelation* hr, SplitKit* kit_sp, gsl_rng* rstate, int n_base
    )


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


def RebuildHaloKit(*rebuild_args):
    return HaloKit(None, rebuild_args = rebuild_args)


cdef class HaloKit:
    cdef:
        HaloRelation hr
        tuple rebuild_args

        public double m_min
        public double m_max


    def __reduce__(self):
        return (RebuildHaloKit, self.rebuild_args)


    def __cinit__(self, hmf, z_grid=None, save=None, rebuild_args=None):
        if rebuild_args is None:
            hmf.update(z = 0)
            self.m_min = 10**hmf.Mmin
            self.m_max = 10**hmf.Mmax
            # Compute sigma_m
            m_arr = np.copy(hmf.m)
            sigma_arr = np.copy(hmf.sigma)
            # Compute alpha_m
            alpha_arr = np.copy(hmf.alpha)
            # Compute delta_z
            if z_grid is None:
                z_arr = np.linspace(0, 200, 1000)
            else:
                z_arr = np.copy(z_grid)
            gf = GrowthFactor(hmf.cosmo)
            delta_arr = hmf.delta_c/np.array([gf.growth_factor(z) for z in z_arr])
        else:
            self.m_min, self.m_max, m_arr, sigma_arr, alpha_arr, z_arr, delta_arr = rebuild_args
        # Set rebuild args
        self.rebuild_args = (self.m_min, self.m_max, m_arr, sigma_arr, alpha_arr, z_arr, delta_arr)
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
        self.hr = HaloRelation_Init(
            &mv_m_arr[0], &mv_sigma_arr[0], &mv_alpha_arr[0], mv_m_arr.shape[0],
            &mv_z_arr[0], &mv_delta_arr[0], mv_z_arr.shape[0]
        )


    def __dealloc__(self):
        HaloRelation_Free(&self.hr)


cdef class HaloMergeTree:
    cdef:
        NodeVector tree
        object z_arr


    property m_root:
        def __get__(self):
            return self.tree.data[0].mass


    property redshifts:
        def __get__(self):
            return np.array(self.z_arr)


    cdef _CheckMassRange(self, HaloKit halo_kit, m_root, m_res):
        if m_root <= halo_kit.m_min or m_root >= halo_kit.m_max:
            raise ValueError("m_root beyond mass interpolation range")
        if m_res <= halo_kit.m_min or m_res >= halo_kit.m_max:
            raise ValueError("m_res beyond mass interpolation range")


    cdef _ConvertRedshift(self, double z_root, double[::1] z_target):
        z_arr = np.append(z_root, z_target)
        self.SetRedshifts(z_arr)
        node_arr = np.asarray(<Node[:self.tree.n_node]>self.tree.data)
        for i_delta, delta in enumerate(np.unique(node_arr['trace'])):
            trace = node_arr['trace']
            trace[trace == delta] = z_arr[i_delta]


    cdef _InitEmpty(self, int n_base):
        self.tree = NodeVector_Init(n_base)


    cdef _InitParkinson08(self,
        HaloKit halo_kit, double m_root, double z_root, double[::1] z_target, double m_res,
        double G0, double gamma1, double gamma2, double eps1, double eps2,
        RandomStateGSL rstate, int n_base
    ):
        self._CheckMassRange(halo_kit, m_root, m_res)
        # Initialise algorithm parameters
        cdef SplitKit kit_sp = SplitKit_Init(G0, gamma1, gamma2, m_res, eps1, eps2)
        # Create the tree
        self.tree = MergeTree_Grow(
            m_root, z_root, &z_target[0], z_target.shape[0], &halo_kit.hr, &kit_sp,
            rstate.rstate, n_base
        )
        # Free
        SplitKit_Free(&kit_sp);
        # Convert delta to redshift
        self._ConvertRedshift(z_root, z_target)
    

    cdef NodeVector* Pointer(self):
        return &self.tree


    cpdef int Append(self, z, mass):
        return NodeVector_Append(&self.tree, z, mass)


    cpdef void LinkTailIndex(self, idx, idx_tail):
        self.tree.data[idx].idx_tail = idx_tail


    cpdef void LinkNextIndex(self, idx, idx_next):
        self.tree.data[idx].idx_next = idx_next


    def SetRedshifts(self, z_arr):
        if self.z_arr is None:
            self.z_arr = np.sort(z_arr)
        else:
            logger.warning("Redshifts cannot be set once again")


    def GetLevel(self, z, tol = 0.01):
        level = np.argmin(abs(self.redshifts - z))
        z_level = self.redshifts[level]
        if abs(z_level - z) > tol:
            logger.warning("Fail to find a redshift within the tolerance")
        return level


    def MassTransfer(self, transfer_func):
        node_arr = np.asarray(<Node[:self.tree.n_node]>self.tree.data)
        node_arr['mass'] = transfer_func(node_arr['mass'])


    def Halos(self, level = -1):
        # Create a copy of all nodes with modified attribute name
        dtype = [('z', 'f8'), ('mass', 'f8'), ('idx_tail', 'i4'), ('idx_next', 'i4')]
        node_arr = np.asarray(<Node[:self.tree.n_node]>self.tree.data).astype(dtype)
        if level == -1:
            return node_arr
        else:
            return node_arr[node_arr['z'] == self.redshifts[level]]


    def __cinit__(self, method = 'Parkinson08', **params):
        self.z_arr = None
        if method == 'Parkinson08':
            self._InitParkinson08(
                params['halo_kit'], params['m_root'], params['z_root'],
                params['z_target'], params['m_res'],
                params['G0'], params['gamma1'], params['gamma2'], params['eps1'], params['eps2'],
                params['rstate'], params['n_base']
            )
        elif method == 'empty':
            self._InitEmpty(params['n_base'])
        else:
            raise ValueError("Method %s is not understood"%method)


    def __dealloc__(self):
        NodeVector_Free(&self.tree)


def Tree_Parkinson08(
    halo_kit, m_root, z_root, z_target, m_res = 1e8, G0 = 0.57, gamma1 = 0.38,
    gamma2 = -0.01, eps1 = 0.1, eps2 = 0.1, rstate = None, n_base = 100000
):
    z_target = np.ravel(z_target).astype('f8')
    if rstate is None:
        rstate = RandomStateGSL(209204)
    return HaloMergeTree(
        method = 'Parkinson08', halo_kit = halo_kit, m_root = m_root, z_root = z_root,
        z_target = z_target, m_res = m_res, G0 = G0, gamma1 = gamma1, gamma2 = gamma2,
        eps1 = eps1, eps2 = eps2, rstate = rstate, n_base = n_base
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                               #
# Functions to analyse trees                                                    #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def EstimateConditionalMassFunction(tree, level, frac_bins = None, mean = True):
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
            m_prog = tree_single.Halos(level_single)['mass']
            for i_bin, (l, u) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
                cmf[i_tree, i_level, i_bin] = np.sum(m_prog[(m_prog > l) & (m_prog < u)])/m_root
    #
    if mean:
        cmf = cmf.mean(axis = 0)
    return frac_mid, cmf/dlnfrac


cdef object _TrackMassAssemblyHistory(Node* node, int n_time):
    cdef:
        double[::1] mah = np.zeros(n_time)
        int i_time
        int idx = 0

    for i_time in range(n_time):
        mah[i_time] = node[idx].mass
        idx = node[idx].idx_tail

    return np.asarray(mah)


def EstimateMassAssemblyHistory(tree):
    tree = np.ravel(tree)
    mah = np.empty(len(tree), dtype = object)
    for i_tree, tree_single in enumerate(tree):
        mah[i_tree] = _TrackMassAssemblyHistory(
            (<HaloMergeTree>tree_single).Pointer().data, len(tree_single.redshifts)
        )/tree_single.m_root
    return mah

