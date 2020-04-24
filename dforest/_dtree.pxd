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

    cdef cppclass KitSplit:
        KitSplit(
            double a0, double a1, double a2,
            double a3, double a4, double a5
        )
        double g0() const
        double gamma1() const
        double gamma2() const
        double m_res() const
        double eps1() const
        double eps2() const
        double j_func(double u_res) const

    cdef cppclass Leaf:
        Leaf(double m, int lvl, int i_t, int i_n)
        double mass;
        int level;
        int idx_tail;
        int idx_next;

    cdef cppclass BinarySplit:
        BinarySplit(
            const HaloRelation* hr, const KitSplit* kit_sp, gsl_rng* rstate
        )
        vector[Leaf] grow(
            double m_root, double z_root, const vector[double]& z_target
        )
