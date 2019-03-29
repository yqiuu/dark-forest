#include"planter.h"

#define SQRT2 sqrt(2.)
#define SQRT2OPI sqrt(2./M_PI)


void HaloRelation_Init(
    HaloRelation *hr, double *mass, double *sigma, double *sigmasq, double *alpha, int n_mass,
    double *z, double *delta_c, int n_z
) {
    hr->sigma_acc = gsl_interp_accel_alloc();
    hr->sigma = gsl_spline_alloc(gsl_interp_cspline, n_mass);
    gsl_spline_init(hr->sigma, mass, sigma, n_mass);

    hr->sigmasq_acc = gsl_interp_accel_alloc();
    hr->sigmasq = gsl_spline_alloc(gsl_interp_cspline, n_mass);
    gsl_spline_init(hr->sigmasq, mass, sigmasq, n_mass);

    hr->alpha_acc = gsl_interp_accel_alloc();
    hr->alpha = gsl_spline_alloc(gsl_interp_cspline, n_mass);
    gsl_spline_init(hr->alpha, mass, alpha, n_mass);

    hr->Z2D_acc = gsl_interp_accel_alloc();
    hr->Z2D = gsl_spline_alloc(gsl_interp_cspline, n_z);
    gsl_spline_init(hr->Z2D, z, delta_c, n_z);

    hr->D2Z_acc = gsl_interp_accel_alloc();
    hr->D2Z = gsl_spline_alloc(gsl_interp_cspline, n_z);
    gsl_spline_init(hr->D2Z, delta_c, z, n_z);
}


void HaloRelation_Free(HaloRelation *hr) {
    gsl_spline_free(hr->sigma);
    gsl_spline_free(hr->sigmasq);
    gsl_spline_free(hr->alpha);
    gsl_interp_accel_free(hr->sigma_acc);
    gsl_interp_accel_free(hr->sigmasq_acc);
    gsl_interp_accel_free(hr->alpha_acc);

    gsl_spline_free(hr->Z2D);
    gsl_spline_free(hr->D2Z);
    gsl_interp_accel_free(hr->Z2D_acc);
    gsl_interp_accel_free(hr->D2Z_acc);
}


void BinaryMergeTree_Alloc_(BinaryMergeTree* tree, int n_base) {
    tree->n_node = 0;
    tree->n_max = n_base;
    tree->node_arr = (BinaryNode*)malloc(n_base*sizeof(BinaryNode));
}


int BinaryMergeTree_NewNode_(BinaryMergeTree *tree, double delta, double mass) {
    if (tree->n_node == tree->n_max) {
        tree->n_max *= 2;
        tree->node_arr = realloc(tree->node_arr, tree->n_max*sizeof(BinaryNode));
    }
    int idx = (tree->n_node)++;
    BinaryNode* node = tree->node_arr + idx;
    node->delta = delta;
    node->mass = mass;
    node->left = -1;
    node->right = -1;
    return idx;
}


void BinaryMergeTree_Trace_Progenitor_(
    BinaryNode *root, int idx, double d_lower, double d_upper,
    int n_max, BinaryNode *out, int *n_target
) {
    BinaryNode *node = root + idx;
    double delta = node->delta;
    double mass = node->mass;
    if (delta > d_lower && delta < d_upper && *n_target < n_max) {
        BinaryNode *target = out + *n_target;
        target->delta = delta;
        target->mass = mass;
        target->left = -1;
        target->right = -1;
        ++(*n_target);
    }
    else {
        if (node->left != -1)
            BinaryMergeTree_Trace_Progenitor_(
                root, node->left, d_lower, d_upper, n_max, out, n_target
            );
        if (node->right != -1)
            BinaryMergeTree_Trace_Progenitor_(
                root, node->right, d_lower, d_upper, n_max, out, n_target
            );
    }
}


int BinaryMergeTree_Nodes(
    BinaryMergeTree *tree, double z, double dz, BinaryNode *node_arr, int n_max
) {
    HaloRelation *hr = tree->hr;
    double d_lower = gsl_spline_eval(hr->Z2D, z, hr->Z2D_acc);
    double d_upper = gsl_spline_eval(hr->Z2D, z + dz, hr->Z2D_acc);
    int n_target = 0;
    BinaryMergeTree_Trace_Progenitor_(
        tree->node_arr, 0, d_lower, d_upper, n_max, node_arr, &n_target
    );
    return n_target;
}


void BinaryMergeTree_Init(
    BinaryMergeTree* tree, HaloRelation* hr, double m_root, double z_root, double z_target,
    double m_res, double eps1, double eps2, int n_base, gsl_rng* rstate
) { 
    //
    tree->hr = hr;
    gsl_interp_accel *sigma_acc = hr->sigma_acc;
    gsl_interp_accel *sigmasq_acc = hr->sigmasq_acc;
    gsl_interp_accel *alpha_acc = hr->alpha_acc;
    gsl_spline *sigma = hr->sigma;
    gsl_spline *sigmasq = hr->sigmasq;
    gsl_spline *alpha = hr->alpha;

    gsl_interp_accel *Z2D_acc = hr->Z2D_acc;
    gsl_spline *Z2D = hr->Z2D;
    //
    BinaryMergeTree_Alloc_(tree, n_base);
    // Create root node
    BinaryMergeTree_NewNode_(tree, gsl_spline_eval(Z2D, z_root, Z2D_acc), m_root);
    // Plant tree
    int idx = 0;
    double d_start;
    double d_target = gsl_spline_eval(Z2D, z_target, Z2D_acc);
    double q_res;
    double m_start, m_half;
    double sigma_start, sigmasq_start, sigmasq_half, sigmasq_res;
    double alpha_half;
    double V_res, V_half;
    double B, beta, eta, pow_diff;
    double dd1, dd2, dd, dN_dd, N_upper;
    double u_res, F;
    double q, m_q, sigmasq_q, R_q, factor1, factor2;
    double m_prog1, m_prog2;
    while(idx < tree->n_node) {
        d_start = (tree->node_arr + idx)->delta;
        m_start = (tree->node_arr + idx)->mass;
        if (d_start >= d_target)
            goto NEXT;
        // Compute useful quantites
        m_half = 0.5*m_start;
        sigma_start = gsl_spline_eval(sigma, m_start, sigma_acc);
        sigmasq_start = gsl_spline_eval(sigmasq, m_start, sigmasq_acc);
        sigmasq_half = gsl_spline_eval(sigmasq, m_half, sigmasq_acc);
        sigmasq_res = gsl_spline_eval(sigmasq, m_res, sigmasq_acc);
        //
        q = 0.;
        q_res = m_res/m_start;
        if (q_res >= 0.5) {
            // No split
            dd = eps1*SQRT2*sqrt(sigmasq_half - sigmasq_start);
            u_res = sigma_start*pow(sigmasq_res - sigmasq_start, -.5);
            F = SQRT2OPI*u_res/sigma_start*dd;
        }
        else {
            // Compute B and beta
            V_res = sigmasq_res*pow(sigmasq_res - sigmasq_start, -1.5);
            V_half = sigmasq_half*pow(sigmasq_half - sigmasq_start, -1.5);
            beta = log(V_res/V_half)/log(2.*q_res);
            B = pow(2., beta)*V_half;
            // Compute ddelta and N_upper
            dd1 = eps1*SQRT2*sqrt(sigmasq_half - sigmasq_start);
            eta = beta - 1;
            pow_diff = pow(2., -eta) - pow(q_res, eta);
            alpha_half = gsl_spline_eval(alpha, m_half, alpha_acc);
            dN_dd = SQRT2OPI*B*pow_diff/eta*alpha_half;
            dd2 = eps2/dN_dd;
            if (dd1 < dd2) {
                dd = dd1;
                N_upper = dN_dd*dd;
            }
            else {
                dd = dd2;
                N_upper = eps2;
            }
            // Compute F
            u_res = sigma_start*pow(sigmasq_res - sigmasq_start, -.5);
            F = SQRT2OPI*u_res/sigma_start*dd;
            // Generate random numbers and split the tree
            if (gsl_rng_uniform(rstate) < N_upper) {
                q = pow(pow(q_res, eta) + pow_diff*gsl_rng_uniform(rstate), 1./eta);
                m_q = q*m_start;
                sigmasq_q = gsl_spline_eval(sigmasq, m_q, sigmasq_acc);
                factor1 = gsl_spline_eval(alpha, m_q, alpha_acc)/alpha_half;
                factor2 = sigmasq_q*pow(sigmasq_q - sigmasq_start, -1.5)/(B*pow(q, beta));
                R_q = factor1*factor2;
                if (gsl_rng_uniform(rstate) > R_q)
                    q = 0.; // No split
            }
        }
        // Set progenitor mass
        if (1 - F - q > q) {
            m_prog1 = (1 - F - q)*m_start;
            m_prog2 = q*m_start;
        }
        else {
            m_prog1 = q*m_start;
            m_prog2 = (1 - F - q)*m_start;
        }
        // Add progenitors
        if (m_prog1 > m_res) {
            // This operation will realloc `tree->node_arr` when necessary
            int left = BinaryMergeTree_NewNode_(tree, d_start + dd, m_prog1);
            (tree->node_arr + idx)->left = left;
        }
        if (m_prog2 > m_res) {
            // This operation will realloc `tree->node_arr` when necessary
            int right = BinaryMergeTree_NewNode_(tree, d_start + dd, m_prog2);
            (tree->node_arr + idx)->right = right;
        }
        //
        NEXT: ++idx;
    }
}


void BinaryMergeTree_Free(BinaryMergeTree* tree) {
    free(tree->node_arr);
}
