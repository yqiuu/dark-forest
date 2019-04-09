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
        HaloRelation *hr, double *mass, double *sigma, double *alpha, int n_mass,
        double *z, double *delta_c, int n_z
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
        double delta
        double mass
        int idx_tail
        int idx_next

    ctypedef struct NodeVector:
        int n_max
        int n_node
        Node* data

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
            &self.hr, &m[0], &sigma[0], &alpha[0], m.shape[0],
            &z_mv[0], &delta_mv[0], z_mv.shape[0]
        )


    def __dealloc__(self):
        HaloRelation_Free(&self.hr)


cdef class HaloMergeTree:
    cdef:
        NodeVector tree
        HaloKit halo_kit
        gsl_rng* rstate


    def Nodes(self, level = -1):
        cdef:
            int i_node
            int n_node = self.tree.n_node
            Node* it_node = self.tree.data
        dtype = np.dtype([('z', 'f8'), ('mass', 'f8'), ('idx_tail', 'i4'), ('idx_next', 'i4')])
        nodes = np.zeros(n_node, dtype = dtype)
        for i_node in range(n_node):
            nodes[i_node]['z'] = it_node.delta
            nodes[i_node]['mass'] = it_node.mass
            nodes[i_node]['idx_tail'] = it_node.idx_tail
            nodes[i_node]['idx_next'] = it_node.idx_next
            it_node += 1
        nodes['z'] = self.halo_kit.D2Z(nodes['z'])
        if level == -1:
            return nodes
        else:
            return nodes[nodes['z'] == np.unique(nodes['z'])[level]]


    def __cinit__(
        self, m_root, z_root, z_target, halo_kit, G0 = 1., gamma1 = 0., gamma2 = 0.,
        m_res = 1e8, eps1 = 0.1, eps2 = 0.2, n_base = 10**5, seed = 2092
    ):
        if m_root <= halo_kit.m_min or m_root >= halo_kit.m_max:
            raise ValueError("Mass of root halo beyond mass interpolation range")
        if m_res <= halo_kit.m_min or m_res >= halo_kit.m_max:
            raise ValueError("Mass resolution halo beyond mass interpolation range")
        self.halo_kit = halo_kit
        self.rstate = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self.rstate, seed)

        cdef:
            double[::1] z_target_mv = np.ravel(z_target).astype(np.double)
            int n_target = z_target_mv.shape[0]
            SplitKit kit_sp = SplitKit_Init(G0, gamma1, gamma2, m_res, eps1, eps2)
        self.tree = MergeTree_Grow(
            m_root, z_root, &z_target_mv[0], n_target, &self.halo_kit.hr, &kit_sp, self.rstate
        )
        SplitKit_Free(&kit_sp);


    def __dealloc__(self):
        NodeVector_Free(&self.tree)
        gsl_rng_free(self.rstate)


