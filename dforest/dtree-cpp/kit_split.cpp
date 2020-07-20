#include"dtree.h"
#include <gsl/gsl_integration.h>

#define BREAK_POINT_1 1e-3
#define BREAK_POINT_2 10.

using namespace std;
using namespace dtree;

static double integrand(double u, void* gamma1) {
    return pow(1. + 1./(u*u), *(double*)gamma1/2.);
}


KitSplit::KitSplit(
    double a0, double a1, double a2, double a3, double a4, double a5
):
    g0_(a0), gamma1_(a1), gamma2_(a2), m_res_(a3), eps1_(a4), eps2_(a5)
{
    if (gamma1_ < 1.) {
        const int n_spl = 25;
        const double du = pow(BREAK_POINT_2/BREAK_POINT_1, 1./(n_spl - 1));
        const int n_intg = 100;
        const int key = 1;
        const double eps_abs = 1e-8;
        const double eps_rel = 1e-5;
        // Construct u_spl
        double u_spl[n_spl];
        double u_iter = BREAK_POINT_1;
        for(int i = 0; i < n_spl - 1; ++i) {
            u_spl[i] = u_iter;
            u_iter *= du;
        }
        u_spl[n_spl - 1] = BREAK_POINT_2;
        // Construct j_u_spl
        double j_u_spl[n_spl];
        double res, err;
        double j_bp1 = pow(BREAK_POINT_1, 1. - gamma1_)/(1. - gamma1_);
        gsl_integration_workspace *workspace
            = gsl_integration_workspace_alloc(n_intg);
        gsl_function target;
        target.function = &integrand;
        target.params = &gamma1_;

        j_u_spl[0] = j_bp1;
        for(int i = 1; i < n_spl; ++i) {
            gsl_integration_qag(
                &target, BREAK_POINT_1, u_spl[i], eps_abs, eps_rel,
                n_intg, key, workspace, &res, &err
            );
            j_u_spl[i] = j_bp1 + res;

        }
        gsl_integration_workspace_free(workspace);
        // Set attributes
        j_bp2_ = j_u_spl[n_spl - 1];
        j_u_acc_ = gsl_interp_accel_alloc();
        j_u_ = gsl_spline_alloc(gsl_interp_cspline, n_spl);
        gsl_spline_init(j_u_, u_spl, j_u_spl, n_spl);
    }
    else {
        cout << "Error: gamma1 should be smaller than 1" << endl;
        exit(EXIT_FAILURE);
    }
}


KitSplit::~KitSplit()
{
    gsl_interp_accel_free(j_u_acc_);
    gsl_spline_free(j_u_);
}


double KitSplit::j_func(double u_res) const
{
    if (gamma1_ == 0.)
        return u_res;
    if (u_res <= BREAK_POINT_1)
        // Use Taylor expansion
        return pow(u_res, 1. - gamma1_)/(1. - gamma1_);
    else if (u_res < BREAK_POINT_2)
        // Use interpolation
        return gsl_spline_eval(j_u_, u_res, j_u_acc_);
    else
        // Use asymptotic expansion
        return j_bp2_ + u_res - BREAK_POINT_2
            - 0.5*gamma1_*(1./u_res - 1./BREAK_POINT_2);
}
