import numpy as np

from libc.stdlib cimport malloc, free

from scipy.interpolate import InterpolatedUnivariateSpline as spl
from hmf.growth_factor import GrowthFactor


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

    void HaloRelation_Init(
        HaloRelation *hr, double *mass, double *sigma, double *sigmasq, double *alpha, int n_mass,
        double *z, double *delta_c, int n_z
    )
    void HaloRelation_Free(HaloRelation *hr)


cdef extern from "clib/halotree.h":
    ctypedef struct BinaryNode:
        double delta
        double mass
        int left
        int right
        
    ctypedef struct BinaryMergeTree:
        int n_node
        BinaryNode* node_arr

    int BinaryMergeTree_Nodes(
        BinaryMergeTree *tree, double z, double dz, BinaryNode *out, int n_max
    )
    void BinaryMergeTree_Init(
        BinaryMergeTree* tree, HaloRelation* hr, double m_root, double z_root, double z_target,
        double m_res, double eps1, double eps2, int n_base, gsl_rng* rstate
    )
    void BinaryMergeTree_Free(BinaryMergeTree* tree)


cdef class HaloRoot:
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
        f = np.sqrt(2./np.pi)*diff_delta*sigmasq_m*diff_sigmasq**-1.5 \
            *np.exp(-.5*diff_delta*diff_delta/diff_sigmasq)*self.alpha(m)
        return f
            

    def __cinit__(self, hmf, z_grid = None):
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
        # Initialisation
        HaloRelation_Init(
            &self.hr, &m[0], &sigma[0], &sigmasq[0], &alpha[0], m.shape[0],
            &z_mv[0], &delta_mv[0], z_mv.shape[0]
        )
        

    def __dealloc__(self):
        HaloRelation_Free(&self.hr)


cdef class HaloMergeTree:
    cdef:
        BinaryMergeTree tree
        HaloRoot halo_root
        gsl_rng* rstate


    def AllNodes(self):
        dtype = np.dtype([('z', 'f8'), ('mass', 'f8'), ('left', 'i4'), ('right', 'i4')])
        delta = np.zeros(self.tree.n_node)
        nodes = np.zeros(self.tree.n_node, dtype = dtype)
        cdef:
            int i_node
            BinaryNode* it_node_arr = self.tree.node_arr
        for i_node in range(self.tree.n_node):
            delta[i_node] = it_node_arr.delta
            nodes[i_node]['mass'] = it_node_arr.mass
            nodes[i_node]['left'] = it_node_arr.left
            nodes[i_node]['right'] = it_node_arr.right
            it_node_arr += 1
        nodes['z'] = self.halo_root.D2Z(delta)
        return nodes


    def Nodes(self, z, dz = 0.1, n_max = 100000):
        cdef:
            BinaryNode* node_arr = <BinaryNode*>malloc(n_max*sizeof(BinaryNode))
            BinaryNode* it_node_arr = node_arr
            int n_node = BinaryMergeTree_Nodes(&self.tree, z, dz, node_arr, n_max)
            int i_node
        dtype = np.dtype([('z', 'f8'), ('mass', 'f8')])
        delta = np.zeros(n_node)
        nodes = np.zeros(n_node, dtype = dtype)
        for i_node in range(n_node):
            delta[i_node] = it_node_arr.delta
            nodes[i_node]['mass'] = it_node_arr.mass
            it_node_arr += 1
        nodes['z'] = self.halo_root.D2Z(delta)
        free(node_arr)
        return nodes


    def __cinit__(
        self, m_root, z_root, z_target, halo_root,
        m_res = 1e8, eps1 = 0.1, eps2 = 0.2, n_base = 10**5, seed = 2092
    ):  
        if m_root <= halo_root.m_min or m_root >= halo_root.m_max:
            raise ValueError("Mass of root halo beyond mass interpolation range")
        if m_res <= halo_root.m_min or m_res >= halo_root.m_max:
            raise ValueError("Mass resolution halo beyond mass interpolation range")
        self.halo_root = halo_root
        self.rstate = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self.rstate, seed)
        BinaryMergeTree_Init(
            &self.tree, &self.halo_root.hr, m_root, z_root, z_target, m_res,
            eps1, eps2, n_base, self.rstate
        )

    
    def __dealloc__(self):
        BinaryMergeTree_Free(&self.tree)
        gsl_rng_free(self.rstate)



