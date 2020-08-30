#ifndef DTREE_H
#define DTREE_H

#include<iostream>
#include<vector>
#include<cmath>
#include<gsl/gsl_errno.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_spline.h>
#include"nut/nut.h"

namespace dtree {
const int kInitialSize = 1000;

class KitSplit {
public:
    KitSplit(double a0, double a1, double a2, double a3, double a4, double a5);
    ~KitSplit();
    double g0() const {return g0_;}
    double gamma1() const {return gamma1_;}
    double gamma2() const {return gamma2_;}
    double m_res() const {return m_res_;}
    double eps1() const {return eps1_;}
    double eps2() const {return eps2_;}
    double j_func(double u_res) const;

private:
    double g0_;
    double gamma1_;
    double gamma2_;
    double m_res_;
    double eps1_;
    double eps2_;

    double j_bp2_; // value of the second break point of J
    gsl_interp_accel *j_u_acc_;
    gsl_spline *j_u_;
};


class HaloRelation {
public:
    HaloRelation(
        double *mass, double *sigma, double *alpha, int n_mass,
        double *z, double *delta_c, int n_z
    );
    HaloRelation(const char* fname);
    ~HaloRelation();
    double eval_sigma(double m) const
        {return gsl_spline_eval(sigma_, m, sigma_acc_);}
    double eval_alpha(double m) const
        {return gsl_spline_eval(alpha_, m, alpha_acc_);}
    double eval_d(double z) const
        {return gsl_spline_eval(Z2D_, z, Z2D_acc_);}
    double eval_z(double d) const
        {return gsl_spline_eval(D2Z_, d, D2Z_acc_);}

private:
    void init_object(
        double *mass, double *sigma, double *alpha, int n_mass,
        double *z, double *delta_c, int n_z
    );

    gsl_interp_accel *sigma_acc_;
    gsl_interp_accel *alpha_acc_;
    gsl_spline *sigma_;
    gsl_spline *alpha_;

    gsl_interp_accel *Z2D_acc_;
    gsl_interp_accel *D2Z_acc_;
    gsl_spline *Z2D_;
    gsl_spline *D2Z_;
};


struct Leaf {
    Leaf(double m, int lvl, int i_t, int i_n)
        : mass(m), level(lvl), idx_tail(i_t), idx_next(i_n) {}
    bool operator< (const Leaf& a) const {return mass >= a.mass;}

    double mass;
    int level;
    int idx_tail;
    int idx_next;
};


using MergerTree = std::vector<Leaf>;

class BinarySplit {
public:
    BinarySplit(
        const HaloRelation* hr, const KitSplit* kit_sp, gsl_rng* rstate
    ) : hr_(hr), kit_sp_(kit_sp), rstate_(rstate) {}
    MergerTree grow(
        double m_root, double z_root, const std::vector<double>& z_target
    );

private:
    int split(
        MergerTree& tree, int idx_root, int level,
        double d_root, double d_target
    );

    const HaloRelation* hr_;
    const KitSplit* kit_sp_;
    gsl_rng* rstate_;
};


// Ensemble method
class ProgMassFunction
{
public:
    ProgMassFunction(const HaloRelation* hr) : hr_(hr) {update(1e12, 0, 0.1);}
    double operator () (double m1)
    {
        if (m1 == m2_)
            return 0.;
        const double sqrt_2opi = sqrt(2/M_PI);
        double sigmasq1 = hr_->eval_sigma(m1);
        sigmasq1 = sigmasq1*sigmasq1;
        double ds = sigmasq1 - sigmasq2_;
        double alpha = -hr_->eval_alpha(m1);
        return sqrt_2opi * m2_/m1/m1 * sigmasq1*dd_*pow(ds, -1.5) \
            * exp(-.5*dd_*dd_/ds)*alpha;
    }
    void update(double m2, double z1, double z2)
    {
        m2_ = m2;
        sigmasq2_ = hr_->eval_sigma(m2);
        sigmasq2_ = sigmasq2_*sigmasq2_;
        dd_ = hr_->eval_d(z1) - hr_->eval_d(z2);
    }

private:
    const HaloRelation* hr_;
    double m2_;
    double sigmasq2_;
    double dd_;
};


class EnsembleGrower
{
public:
    EnsembleGrower(
        ProgMassFunction* prog_func, gsl_rng* rstate, double m_res, int n_draw
    ) : prog_func_(prog_func), rstate_(rstate), m_res_(m_res),
        n_draw_(n_draw), pool_prog_(n_draw) {}
    int grow_level(
        MergerTree& tree, int idx_root, int level,
        double z_target, double z_root
    );
    MergerTree grow_tree(
        double m_root, double z_root, const std::vector<double>& z_target
    );

private:
    void sample(double m2, double z1, double z2);

    ProgMassFunction* prog_func_;
    gsl_rng* rstate_;
    double m_res_;
    int n_draw_;
    std::vector< std::vector<double> > pool_prog_;
};

} // dtree
#endif // DTREE_H
