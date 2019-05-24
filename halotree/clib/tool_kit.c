#include"halotree.h"
#include <gsl/gsl_integration.h>

#define N_WORKSPACE 100
#define EPS_ABS 1e-8
#define EPS_REL 1e-5
#define N_SPLINE 25
#define KEY 1

#define BREAK_POINT_1 1e-3
#define BREAK_POINT_2 10.
#define INTERVAL pow(BREAK_POINT_2/BREAK_POINT_1, 1./(N_SPLINE - 1))


static double Integrand(double u, void * gamma1) {
    return pow(1. + 1./(u*u), *(double*)gamma1/2.);
}


double SplitKit_J(SplitKit *kit_sp, double u_res) {
    double gamma1 = kit_sp->gamma1;
    if (gamma1 == 0.)
        return u_res;
    if (u_res <= BREAK_POINT_1)
        // Use Taylor expansion
        return pow(u_res, 1. - gamma1)/(1. - gamma1);
    else if (u_res < BREAK_POINT_2)
        // Use interpolation
        return gsl_spline_eval(kit_sp->J_u, u_res, kit_sp->J_u_acc);
    else
        // Use asymptotic expansion
        return kit_sp->J_bp2 + u_res - BREAK_POINT_2 - 0.5*gamma1*(1./u_res - 1./BREAK_POINT_2);
}


SplitKit SplitKit_Init(
    double G0, double gamma1, double gamma2, double m_res, double eps1, double eps2
) {
    SplitKit kit_sp;
    kit_sp.G0 = G0;
    kit_sp.gamma1 = gamma1;
    kit_sp.gamma2 = gamma2;
    kit_sp.m_res = m_res;
    kit_sp.eps1 = eps1;
    kit_sp.eps2 = eps2;

    if (gamma1 < 1.) {
        // Construct u_spline
        double u_spline[N_SPLINE];
        double u_iter = BREAK_POINT_1;
        for(int i = 0; i < N_SPLINE - 1; ++i) {
            u_spline[i] = u_iter;
            u_iter *= INTERVAL;
        }
        u_spline[N_SPLINE - 1] = BREAK_POINT_2;
        // Construct J_u_spline
        double J_u_spline[N_SPLINE];
        double res, err;
        double J_bp1 = pow(BREAK_POINT_1, 1. - gamma1)/(1. - gamma1);
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(N_WORKSPACE);
        gsl_function target;
        target.function = &Integrand;
        target.params = &gamma1;

        J_u_spline[0] = J_bp1;
        for(int i = 1; i < N_SPLINE; ++i) {
            gsl_integration_qag(
                &target, BREAK_POINT_1, u_spline[i], EPS_ABS, EPS_REL,
                N_WORKSPACE, KEY, workspace, &res, &err
            );
            J_u_spline[i] = J_bp1 + res;

        }
        gsl_integration_workspace_free(workspace);
        // Set attributes
        kit_sp.J_bp1 = J_bp1;
        kit_sp.J_bp2 = J_u_spline[N_SPLINE - 1];
        kit_sp.J_u_acc = gsl_interp_accel_alloc();
        kit_sp.J_u = gsl_spline_alloc(gsl_interp_cspline, N_SPLINE);
        gsl_spline_init(kit_sp.J_u, u_spline, J_u_spline, N_SPLINE);
    }
    else {
        printf("Error: gamma1 should be smaller than 1");
        exit(EXIT_FAILURE);
    }
    return kit_sp;
}


void SplitKit_Free(SplitKit *kit_sp) {
    gsl_interp_accel_free(kit_sp->J_u_acc);
    gsl_spline_free(kit_sp->J_u);
}


AugmentKit AugmentKit_Init(int n_trail, int n_max_trail, double eps, double m_cut) {
    AugmentKit kit_ag;
    kit_ag.n_trail = n_trail;
    kit_ag.n_max_trail = n_max_trail;
    kit_ag.eps = eps;
    kit_ag.m_cut = m_cut;
    return kit_ag;
}
