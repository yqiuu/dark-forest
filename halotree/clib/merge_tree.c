#include<math.h>
#include"halotree.h"

#define SQRT2 sqrt(2.)
#define SQRT2OPI sqrt(2./M_PI)
#define N_BASE_GROW 1000000
#define INITIAL_FIT -1.


static int CompareMass(const void *a, const void *b) {
    double value = ((Node*)a)->mass - ((Node*)b)->mass;
    if (value > 0)
        return -1;
    else if (value < 0)
        return 1;
    else
        return 0;
}


static inline double ComputeFraction(
    double sigma_start, double sigmasq_start, double sigmasq_res,
    double G1, double dd, SplitKit *kit_sp
) {
    double u_res = sigma_start*pow(sigmasq_res - sigmasq_start, -.5);
    return SQRT2OPI*SplitKit_J(kit_sp, u_res)*G1/sigma_start*dd;
}


static inline int ReduceDelta(double dd_target, double *dd) {
    if (*dd > dd_target) {
        *dd = dd_target;
        return 1;
    }
    else
        return 0;
}


int MergeTree_BinarySplit(
    NodeVector *tree, int idx_root, double d_target,
    HaloRelation* hr, SplitKit* kit_sp, gsl_rng* rstate
) {
    // Load algorithm parameters
    double G0 = kit_sp->G0;
    double gamma1 = kit_sp->gamma1;
    double gamma2 = kit_sp->gamma2;
    double m_res = kit_sp->m_res;
    double eps1 = kit_sp->eps1;
    double eps2 = kit_sp->eps2;
    // Load interpolation tables for sigma_m
    gsl_interp_accel *sigma_acc = hr->sigma_acc;
    gsl_interp_accel *alpha_acc = hr->alpha_acc;
    gsl_spline *sigma = hr->sigma;
    gsl_spline *alpha = hr->alpha;
    // Initialise the binary tree
    NodeVector tree_binary = NodeVector_Init(N_BASE_GROW);
    NodeVector_Append(&tree_binary, tree->data[idx_root].trace, tree->data[idx_root].mass);
    // Plant tree
    double mu, G1, G2;
    double d_start;
    double q_res;
    double m_start, m_half;
    double sigma_start, sigma_half, sigma_res;
    double sigmasq_start, sigmasq_half, sigmasq_res;
    double alpha_half;
    double V_res, V_half;
    double B, beta, eta, pow_diff;
    double dd1, dd2, dd, dd_target, dN_dd, N_upper, F;
    double q, m_q, sigma_q, sigmasq_q, R_q, factor1, factor2;
    double m_prog1, m_prog2;
    int save;
    int n_out = 0;
    int idx_first = -1;
    int idx = 0;
    while(idx < tree_binary.n_node) {
        d_start = tree_binary.data[idx].trace;
        m_start = tree_binary.data[idx].mass;
        dd_target = d_target - d_start;
        save = 0;
        // Compute useful quantites
        m_half = 0.5*m_start;
        sigma_start = gsl_spline_eval(sigma, m_start, sigma_acc);
        sigmasq_start = sigma_start*sigma_start;
        sigma_half = gsl_spline_eval(sigma, m_half, sigma_acc);
        sigmasq_half = sigma_half*sigma_half;
        sigma_res = gsl_spline_eval(sigma, m_res, sigma_acc);
        sigmasq_res = sigma_res*sigma_res;
        G1 = G0*pow(d_start/sigma_start, gamma2);
        //
        q = 0.;
        q_res = m_res/m_start;
        if (q_res >= 0.5) {
            // No split
            dd = eps1*SQRT2*sqrt(sigmasq_half - sigmasq_start);
            save = ReduceDelta(dd_target, &dd);
            F = ComputeFraction(sigma_start, sigmasq_start, sigmasq_res, G1, dd, kit_sp);
        }
        else {
            // Compute B and beta
            V_res = sigmasq_res*pow(sigmasq_res - sigmasq_start, -1.5);
            V_half = sigmasq_half*pow(sigmasq_half - sigmasq_start, -1.5);
            beta = log(V_res/V_half)/log(2.*q_res);
            B = pow(2., beta)*V_half;
            // Compute ddelta1
            dd1 = eps1*SQRT2*sqrt(sigmasq_half - sigmasq_start);
            // Compute ddelta2
            alpha_half = gsl_spline_eval(alpha, m_half, alpha_acc);
            mu = gamma1 < 0. ? -log(sigma_res/sigma_half)/log(2.*q_res) : alpha_half;
            eta = beta - 1 - gamma1*mu;
            pow_diff = pow(.5, eta) - pow(q_res, eta);
            G2 = G1*pow(sigma_half/sigma_start, gamma1)*pow(0.5, mu*gamma1);
            dN_dd = SQRT2OPI*B*pow_diff/eta*alpha_half*G2;
            dd2 = eps2/dN_dd;
            // Choose
            if (dd1 < dd2)
                dd = dd1;
            else
                dd = dd2;
            save = ReduceDelta(dd_target, &dd);
            N_upper = dN_dd*dd;
            // Compute F
            F = ComputeFraction(sigma_start, sigmasq_start, sigmasq_res, G1, dd, kit_sp);
            // Generate random numbers and split the tree
            if (gsl_rng_uniform(rstate) < N_upper) {
                q = pow(pow(q_res, eta) + pow_diff*gsl_rng_uniform(rstate), 1./eta);
                m_q = q*m_start;
                sigma_q = gsl_spline_eval(sigma, m_q, sigma_acc);
                sigmasq_q = sigma_q*sigma_q;
                factor1 = gsl_spline_eval(alpha, m_q, alpha_acc)/alpha_half;
                factor2 = sigmasq_q*pow(sigmasq_q - sigmasq_start, -1.5)/(B*pow(q, beta));
                R_q = factor1*factor2;
                if (gsl_rng_uniform(rstate) > R_q)
                    q = 0.; // No split
            }
        }
        // Compute progenitor mass
        m_prog1 = (1 - F - q)*m_start;
        m_prog2 = q*m_start;
        // Add progenitors
        if (save) {
            if (m_prog1 > m_res) {
                if (n_out == 0)
                    idx_first = NodeVector_Append(tree, d_target, m_prog1);
                else
                    NodeVector_Append(tree, d_target, m_prog1);
                ++n_out;
            }
            if (m_prog2 > m_res) {
                if (n_out == 0)
                    idx_first = NodeVector_Append(tree, d_target, m_prog2);
                else
                    NodeVector_Append(tree, d_target, m_prog2);
                ++n_out;
            }
        }
        else {
            if (m_prog1 > m_res)
                NodeVector_Append(&tree_binary, d_start + dd, m_prog1);
            if (m_prog2 > m_res)
                NodeVector_Append(&tree_binary, d_start + dd, m_prog2);
        }
        //
        ++idx;
    }
    // Link progenitor indices
    if (n_out > 0) {
        qsort(tree->data + idx_first, n_out, sizeof(Node), CompareMass);
        tree->data[idx_root].idx_tail = idx_first;
        for(int i_prog = idx_first; i_prog - idx_first < n_out - 1; ++i_prog)
            tree->data[i_prog].idx_next = i_prog + 1;
    }
    //
    NodeVector_Free(&tree_binary);
    return n_out;
}


NodeVector MergeTree_Grow(
    double m_root, double z_root, double *z_target, int n_target,
    HaloRelation* hr, SplitKit *kit_sp, gsl_rng* rstate
) {
    // Convet redshift to delta_c
    gsl_interp_accel *Z2D_acc = hr->Z2D_acc;
    gsl_spline *Z2D= hr->Z2D;
    double d_root = gsl_spline_eval(Z2D, z_root, Z2D_acc);
    double *delta_target = malloc(n_target*sizeof(double));
    for(int i_d = 0; i_d < n_target; ++i_d)
        delta_target[i_d] = gsl_spline_eval(Z2D, z_target[i_d], Z2D_acc);
    //
    NodeVector tree = NodeVector_Init(N_BASE_GROW);
    NodeVector_Append(&tree, d_root, m_root);
    //
    int idx_root = 0;
    int n_prog;
    int n_prog_next = 1;
    double d_working;
    for(int i_d = 0; i_d < n_target; ++i_d) {
        n_prog = n_prog_next;
        n_prog_next = 0;
        d_working = delta_target[i_d];
        for(int i_prog = 0; i_prog < n_prog; ++i_prog)
            n_prog_next += MergeTree_BinarySplit(&tree, idx_root++, d_working, hr, kit_sp, rstate);
    }
    free(delta_target);
    return tree;
}


static inline double* SelectTrailProgs(Node* nodes, int n_node, double d_trail, int *n_prog) {
    double* m_prog = malloc(n_node*sizeof(double));

    for(int i_node = 0; i_node < n_node; ++i_node) {
        if (nodes[i_node].trace == d_trail)
            m_prog[(*n_prog)++] = nodes[i_node].mass;
    }

    return m_prog;
}


static inline int FailCriterionA(int n_prog, int n_trail_prog) {
    if (n_prog > n_trail_prog)
        return 1;
    else
        return 0;
}


static inline int FailCriterionB(
    double* m_prog, double* m_trail_prog, double n_prog, double eps, double *fit
) {
    int success = 1;
    double eps_trail;
    double fit_trail = 0.;

    for(int i_prog = 0; i_prog < n_prog; ++i_prog) {
        eps_trail = fabs(m_trail_prog[i_prog]/m_prog[i_prog] - 1.);
        if (eps_trail >= eps)
            success = 0;
        fit_trail += eps_trail;
    }
    *fit = fit_trail;

    return !success;
}


static inline int FailCriterionC(
    double* m_trail_prog, int n_trail_prog, int n_prog, double m_cut
) {
    int success = 1;

    for(int i_prog = n_prog; i_prog < n_trail_prog; ++i_prog) {
        if (m_trail_prog[i_prog] >= m_cut)
            success = 0;
    }

    return !success;
}


NodeVector MergeTree_Augment(
    double *m_prog, int n_prog, double m_root, double z_root, double *z_target, int n_target,
    HaloRelation* hr, AugmentKit *kit_ag, SplitKit *kit_sp, gsl_rng* rstate, int n_base
) {
    // Augment a pruned branch of the halo merge tree
    //
    // m_prog should be in descending order

    // Load algorithm parameters
    int i_trail = 1;
    int n_trail = kit_ag->n_trail;
    int n_max_trail = kit_ag->n_max_trail;
    double eps = kit_ag->eps;
    double m_cut = kit_ag->eps;
    // Calculate d_trail
    double d_trail = gsl_spline_eval(hr->Z2D, z_target[n_target - 1], hr->Z2D_acc);
    double *m_trail_prog;
    int n_trail_prog = 0;
    //
    NodeVector tree;
    NodeVector tree_best_fit;
    double fit = INITIAL_FIT;
    double best_fit = INITIAL_FIT;
    int not_match = 1;
    do {
        // Check if N_max_trail is reached
        if (i_trail > n_max_trail) {
            if (best_fit == INITIAL_FIT) {
                printf("Warning: fail to find a match; return an empty tree\n");
                tree = NodeVector_Init(n_base);
            }
            else {
                printf("Warning: fail to find a match; return the best match so far\n");
                tree = tree_best_fit;
            }
            break;
        }
        // Check if N_trail is reached
        if (i_trail%n_trail == 0) {
            eps *= (1 + eps);
            if (best_fit != INITIAL_FIT) {
                tree = tree_best_fit;
                m_trail_prog = NULL;
                goto CRITERION_B;
            }
        }
        // Generate a trail tree
        tree = MergeTree_Grow(m_root, z_root, z_target, n_target, hr, kit_sp, rstate);
        n_trail_prog = 0;
        m_trail_prog = SelectTrailProgs(tree.data, tree.n_node, d_trail, &n_trail_prog);
        //
        if (
            FailCriterionA(n_prog, n_trail_prog) || \
            FailCriterionC(m_trail_prog, n_trail_prog, n_prog, m_cut)
        ) {
            ++i_trail;
            NodeVector_Free(&tree);
        }
        else {
            CRITERION_B:
            if (FailCriterionB(m_prog, m_trail_prog, n_prog, eps, &fit)) {
                ++i_trail;
                // Record the best match
                if (fit < best_fit || best_fit == INITIAL_FIT) {
                    best_fit = fit;
                    tree_best_fit = tree;
                }
                else
                    NodeVector_Free(&tree);
            }
            else
                not_match = 0;
        }
        free(m_trail_prog);
    } while(not_match);

    return tree;
}
