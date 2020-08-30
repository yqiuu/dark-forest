from libcpp.vector cimport vector


cdef extern from "<gsl/gsl_rng.h>":
    ctypedef struct gsl_rng:
        pass

    ctypedef struct gsl_rng_type:
        pass

    gsl_rng_type* gsl_rng_mt19937

    gsl_rng* gsl_rng_alloc(gsl_rng_type *T)
    void gsl_rng_set(gsl_rng *r, unsigned long int s)
    void gsl_rng_free(gsl_rng *r)


cdef extern from "dtree-cpp/dtree.h" namespace "dtree":
    int kInitialSize

    cdef cppclass HaloRelation:
        HaloRelation(
            double* mass, double* sigma, double* alpha, int n_mass,
            double* z, double* delta_c, int n_z
        )
        HaloRelation(const char* fname) except +

    cdef cppclass Leaf:
        Leaf(double m, int lvl, int i_t, int i_n)
        double mass;
        int level;
        int idx_tail;
        int idx_next;

    cdef cppclass BinarySplit:
        BinarySplit(
            const HaloRelation* hr, gsl_rng* rstate, double m_res,
            double g0, double gamma1, double gamma2, double eps1, double eps2
        )
        vector[Leaf] grow(
            double m_root, double z_root, const vector[double]& z_target
        )

    cdef cppclass EnsembleGrower:
        EnsembleGrower(
            const HaloRelation* hr,gsl_rng* rstate,
            double m_res, double m_eps, int n_draw
        )
        vector[Leaf] grow_tree(
            double m_root, double z_root, const vector[double]& z_target
        )
