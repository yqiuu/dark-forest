from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

import logging
import struct as binary

import numpy as np, h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf.growth_factor import GrowthFactor
from hmf.hmf import MassFunction


def CreateLogger():
    logger = logging.getLogger(__name__)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s:%(message)s'))
    logger.addHandler(stream_handler)
    return logger
logger = CreateLogger()


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
        HaloRelation* hr, SplitKit* kit_sp, gsl_rng* rstate
    )


cdef class HaloKit:
    cdef:
        HaloRelation hr

        public object sigma
        public object sigmasq
        public object alpha
        public object Z2D
        public object D2Z

        public double m_min
        public double m_max


    def ConditionalMassFunction(self, m, z, m_root, z_root):
        sigmasq_m = self.sigmasq(m)
        sigmasq_root = self.sigmasq(m_root)
        diff_delta = self.Z2D(z) - self.Z2D(z_root)
        diff_sigmasq = sigmasq_m - sigmasq_root
        factor_exp = np.exp(-.5*diff_delta*diff_delta/diff_sigmasq)
        return sqrt(2./np.pi)*diff_delta*sigmasq_m*diff_sigmasq**-1.5*factor_exp*self.alpha(m)


    def ProgenitorMassFunction(self, m, z, m_root, z_root):
        return m_root/m*self.ConditionalMassFunction(m, z, m_root, z_root)


    def __cinit__(self, hmf, z_grid = None, save = None):
        hmf.update(z = 0)
        self.m_min = 10**hmf.Mmin
        self.m_max = 10**hmf.Mmax
        # Compute sigma_m
        cdef:
            double[::1] m = np.copy(hmf.m)
            double[::1] sigma = np.copy(hmf.sigma)
            double[::1] sigmasq = np.copy(hmf.sigma*hmf.sigma)
        self.sigma = spl(hmf.m, hmf.sigma)
        self.sigmasq = spl(hmf.m, hmf.sigma*hmf.sigma)
        # Compute alpha_m
        logm = np.log(hmf.m)
        lnsigma = spl(logm, hmf.lnsigma)
        cdef:
            double[::1] alpha = lnsigma.derivative(n = 1)(logm)
        self.alpha = spl(hmf.m, np.asarray(alpha))
        # Compute delta_z
        if z_grid is None:
            z_grid = np.linspace(0, 200, 1000)
        else:
            z_grid = np.copy(z_grid)
        delta_c0 = hmf.delta_c
        gf = GrowthFactor(hmf.cosmo)
        delta_c = delta_c0/np.array([gf.growth_factor(z) for z in z_grid])
        self.Z2D = spl(z_grid, delta_c)
        self.D2Z = spl(delta_c, z_grid)
        cdef:
            double[::1] z_mv = z_grid
            double[::1] delta_mv = delta_c
        # Save a binary file of necessary arrays for external uses
        if save is not None:
            with open(save, "wb") as fp:
                n_mass = m.shape[0]
                fp.write(binary.pack('i', n_mass))
                fp.write(binary.pack('%dd'%n_mass, *np.asarray(m)))
                fp.write(binary.pack('%dd'%n_mass, *np.asarray(sigma)))
                fp.write(binary.pack('%dd'%n_mass, *np.asarray(alpha)))

                n_z = z_mv.shape[0]
                fp.write(binary.pack('i', n_z))
                fp.write(binary.pack('%dd'%n_z, *z_grid))
                fp.write(binary.pack('%dd'%n_z, *delta_c))
        # Initialisation
        self.hr = HaloRelation_Init(
            &m[0], &sigma[0], &alpha[0], m.shape[0], &z_mv[0], &delta_mv[0], z_mv.shape[0]
        )


    def __dealloc__(self):
        HaloRelation_Free(&self.hr)


cdef class HaloMergeTree:
    cdef:
        NodeVector tree
        object redshifts


    property m_root:
        def __get__(self):
            return self.tree.data[0].mass


    property redshifts:
        def __get__(self):
            return np.array(self.redshifts)


    cdef NodeVector* pointer(self):
        return &self.tree


    def Nodes(self, level = -1):
        # Create a copy of all nodes with modified attribute name
        dtype = [('z', 'f8'), ('mass', 'f8'), ('idx_tail', 'i4'), ('idx_next', 'i4')]
        node_arr = np.asarray(<Node[:self.tree.n_node]>self.tree.data).astype(dtype)
        if level == -1:
            return node_arr
        else:
            return node_arr[node_arr['z'] == np.unique(node_arr['z'])[level]]


    def __cinit__(self, method = 'Parkinson08', **params):
        cdef:
            gsl_rng *rstate
            double[::1] z_target_mv
            HaloKit halo_kit
            SplitKit kit_sp

            Node[::1] node_mv

        if method == 'empty':
            self.tree = NodeVector_Init(params['n_base'])
        elif method == 'Parkinson08':
            params_method = {
                'm_root':1e12, 'z_root':0., 'z_target':1.,
                'G0':1., 'gamma1':0., 'gamma2':0.,
                'm_res':1e8, 'eps1':0.1, 'eps2':0.1, 'seed':2092
            }
            params_method.update(params)
            halo_kit = params_method['halo_kit']
            m_root = params_method['m_root']
            z_root = params_method['z_root']
            z_target = params_method['z_target']
            G0 = params_method['G0']
            gamma1 = params_method['gamma1']
            gamma2 = params_method['gamma2']
            m_res = params_method['m_res']
            eps1 = params_method['eps1']
            eps2 = params_method['eps2']
            seed = params_method['seed']
            #
            if m_root <= halo_kit.m_min or m_root >= halo_kit.m_max:
                raise ValueError("Mass of root halo beyond mass interpolation range")
            if m_res <= halo_kit.m_min or m_res >= halo_kit.m_max:
                raise ValueError("Mass resolution halo beyond mass interpolation range")
            # Create random number generator
            rstate = gsl_rng_alloc(gsl_rng_mt19937)
            gsl_rng_set(rstate, seed)
            # Create the tree
            z_target_mv = np.ravel(z_target).astype(np.double)
            n_target = z_target_mv.shape[0]
            kit_sp = SplitKit_Init(G0, gamma1, gamma2, m_res, eps1, eps2)
            self.tree = MergeTree_Grow(
                m_root, z_root, &z_target_mv[0], n_target, &halo_kit.hr, &kit_sp, rstate
            )
            SplitKit_Free(&kit_sp);
            gsl_rng_free(rstate)
            # Convert delta to redshift
            node_arr = np.asarray(<Node[:self.tree.n_node]>self.tree.data)
            redshifts = halo_kit.D2Z(node_arr['trace'])
            node_arr['trace'] = redshifts
            self.redshifts = np.unique(redshifts)
        else:
            raise ValueError("Method %s is not understood"%method)


    def __dealloc__(self):
        NodeVector_Free(&self.tree)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                               #
# Functions to analyse trees                                                    #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def EstimateConditionalMassFunction(tree, level, frac_bins = None):
    tree = np.ravel(tree)
    level = np.ravel(level)
    # Set fraction bins
    if frac_bins is None:
        frac_bins = np.logspace(-4, 0, 21)
    lnfrac_bins = np.log(frac_bins)
    dlnfrac = np.diff(lnfrac_bins)
    frac_mid = np.exp(lnfrac_bins[1:] + .5*dlnfrac)
    #
    cmf = np.zeros([len(tree), len(level), len(frac_mid)])
    for i_tree, tree_single in enumerate(tree):
        m_root = tree_single.m_root
        mass_bins = frac_bins*m_root
        for i_level, level_single in enumerate(level):
            m_prog = tree_single.Nodes(level_single)['mass']
            for i_bin, (l, u) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
                cmf[i_tree, i_level, i_bin] = np.sum(m_prog[(m_prog > l) & (m_prog < u)])/m_root
    return frac_mid, cmf/dlnfrac


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                               #
# Functions to read VELOCIraptor trees                                          #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
DEF N_MAX_HALO = 10**12


cdef inline tuple _ID2S(long i):
    """
    Convert halo ID to snapshot and index respectively.
    """
    return (i//N_MAX_HALO, i%N_MAX_HALO)


def ReadVELOCIraptor(fname, snap_min, snap_max, m_min, m_max, key_mass = 'Mass_200crit'):
    # Open the file
    fp = h5py.File(fname, "r")
    # Read Cosmology parameters
    attrs = fp.get('Header')['Simulation'].attrs
    h = attrs['h_val']
    Om0 = attrs['Omega_m']
    halo_kit = HaloKit(
        MassFunction(Mmin = 3, dlog10m = 0.05, cosmo_params = {'H0':100.*h, 'Om0':Om0})
    )
    # Convert little h
    m_min /= 1e10*h
    m_max /= 1e10*h
    # Read redshifts
    redshifts = np.zeros(snap_max + 1)
    for snap in range(snap_max + 1):
        redshifts[snap] = 1./fp.get('Snap_%03d'%snap).attrs['scalefactor'] - 1.
    # Select and initialise root halos
    cdef:
        NodeVector* tree_ptr = NULL

    grp = fp.get('Snap_%03d'%snap_max)
    mass = np.asarray(grp[key_mass])
    cond = (mass > m_min) & (mass < m_max)
    mass = mass[cond]
    tail = np.asarray(grp['Tail'])[cond]

    n_gal = len(mass)
    tree_arr = np.empty(n_gal, dtype = object)
    pool_arr = np.empty(n_gal, dtype = object)

    logger.info("Read %d root halos at given mass range"%n_gal)
    for i_gal in range(n_gal):
        tree = HaloMergeTree(halo_kit, method = 'empty', n_base = 10**5)
        tree_arr[i_gal] = tree

        pool = {}
        idx_node_new = NodeVector_Append(tree.pointer(), redshifts[snap_max], mass[i_gal])
        pool[idx_node_new] = _ID2S(tail[i_gal])
        pool_arr[i_gal] = pool
    # Read progentiros
    for snap in range(snap_max - 1, snap_min - 1, -1):
        logger.info("Read progenitors at snapshot %d"%snap)
        z = redshifts[snap]
        grp = fp.get('Snap_%03d'%snap)
        mass = np.asarray(grp[key_mass])
        tail = np.asarray(grp['Tail'])
        prog_next = np.asarray(grp['NextProgenitor'])
        for i_gal in range(n_gal):
            tree_ptr = (<HaloMergeTree>tree_arr[i_gal]).pointer()
            pool = pool_arr[i_gal]
            for idx_node_desc in list(pool):
                snap_tail, idx_tail = pool[idx_node_desc]
                if snap == snap_tail:
                    idx_node_new = NodeVector_Append(tree_ptr, z, mass[idx_tail])
                    tree_ptr.data[idx_node_desc].idx_tail = idx_node_new
                    # Read next progenitors
                    idx_node_prev = idx_node_new
                    idx_prev = idx_tail
                    while True:
                        snap_next, idx_next = _ID2S(prog_next[idx_prev])
                        if idx_next != idx_prev and snap == snap_next:
                            idx_node_new = NodeVector_Append(tree_ptr, z, mass[idx_next])
                            tree_ptr.data[idx_node_prev].idx_next = idx_node_new
                            idx_node_prev = idx_node_new
                            idx_prev = idx_next
                        else:
                            break
                    del pool[idx_node_desc]
    dtype = [('z', 'f8'), ('mass', 'f8'), ('idx_tail', 'i4'), ('idx_next', 'i4')]
    for i_gal in range(n_gal):
        tree_arr[i_gal]._build_array()
    # Close the file
    fp.close()
    return tree_arr
