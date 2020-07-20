#include<algorithm>
#include"dtree.h"

using namespace std;
using namespace dtree;

#define SQRT2 sqrt(2.)
#define SQRT2OPI sqrt(2./M_PI)
#define INITIAL_FIT -1.

struct Node {
    Node(double m=0., double d=0.) : mass(m), delta(d) {}
    double mass;
    double delta;
};


static inline void append_leaf(MergerTree& tree, double m, int lvl)
{
    tree.push_back(Leaf(m, lvl, -1, -1));
}


static inline void append_progenitor(
    MergerTree& tree, int& n_out, int& idx_first, double m, int lvl
)
{
    append_leaf(tree, m, lvl);
    if (n_out == 0)
        idx_first = tree.size() - 1;
    ++n_out;
}


static inline double compute_frac(
    double sigma_start, double sigmasq_start, double sigmasq_res,
    double g1, double dd, const KitSplit* kit_sp
) {
    double u_res = sigma_start*pow(sigmasq_res - sigmasq_start, -.5);
    return SQRT2OPI*kit_sp->j_func(u_res)*g1/sigma_start*dd;
}


static inline int reduce_delta(double& dd, double dd_target) {
    if (dd > dd_target) {
        dd = dd_target;
        return 1;
    }
    else
        return 0;
}


int BinarySplit::split(
    MergerTree& tree, int idx_root, int level, double d_root, double d_target
)
{
    // Load algorithm parameters
    const double g0 = kit_sp_->g0();
    const double gamma1 = kit_sp_->gamma1();
    const double gamma2 = kit_sp_->gamma2();
    const double m_res = kit_sp_->m_res();
    const double eps1 = kit_sp_->eps1();
    const double eps2 = kit_sp_->eps2();
    // Initialise the binary tree
    vector<Node> tree_binary;
    tree_binary.reserve(kInitialSize);
    tree_binary.push_back(Node(tree[idx_root].mass, d_root));
    // grow tree
    double mu, g1, g2;
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
    while(idx < tree_binary.size()) {
        m_start = tree_binary[idx].mass;
        d_start = tree_binary[idx].delta;
        dd_target = d_target - d_start;
        save = 0;
        // Compute useful quantites
        m_half = 0.5*m_start;
        sigma_start = hr_->eval_sigma(m_start);
        sigmasq_start = sigma_start*sigma_start;
        sigma_half = hr_->eval_sigma(m_half);
        sigmasq_half = sigma_half*sigma_half;
        sigma_res = hr_->eval_sigma(m_res);
        sigmasq_res = sigma_res*sigma_res;
        g1 = g0*pow(d_start/sigma_start, gamma2);
        //
        q = 0.;
        q_res = m_res/m_start;
        if (q_res >= 0.5) {
            // No split
            dd = eps1*SQRT2*sqrt(sigmasq_half - sigmasq_start);
            save = reduce_delta(dd, dd_target);
            F = compute_frac(
                sigma_start, sigmasq_start, sigmasq_res, g1, dd, kit_sp_
            );
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
            alpha_half = hr_->eval_alpha(m_half);
            mu = gamma1 < 0. ?
                -log(sigma_res/sigma_half)/log(2.*q_res) : alpha_half;
            eta = beta - 1 - gamma1*mu;
            pow_diff = pow(.5, eta) - pow(q_res, eta);
            g2 = g1*pow(sigma_half/sigma_start, gamma1)*pow(0.5, mu*gamma1);
            dN_dd = SQRT2OPI*B*pow_diff/eta*alpha_half*g2;
            dd2 = eps2/dN_dd;
            // Choose
            if (dd1 < dd2)
                dd = dd1;
            else
                dd = dd2;
            save = reduce_delta(dd, dd_target);
            N_upper = dN_dd*dd;
            //
            F = compute_frac(
                sigma_start, sigmasq_start, sigmasq_res, g1, dd, kit_sp_
            );
            // Generate random numbers and split the tree
            if (gsl_rng_uniform(rstate_) < N_upper) {
                q = pow(pow(q_res, eta)
                    + pow_diff*gsl_rng_uniform(rstate_), 1./eta);
                m_q = q*m_start;
                sigma_q = hr_->eval_sigma(m_q);
                sigmasq_q = sigma_q*sigma_q;
                factor1 = hr_->eval_alpha(m_q)/alpha_half;
                factor2 = sigmasq_q*pow(sigmasq_q - sigmasq_start, -1.5)
                    /(B*pow(q, beta));
                R_q = factor1*factor2;
                if (gsl_rng_uniform(rstate_) > R_q)
                    q = 0.; // No split
            }
        }
        // Compute progenitor mass
        m_prog1 = (1 - F - q)*m_start;
        m_prog2 = q*m_start;
        // Add progenitors
        if (save) {
            if (m_prog1 > m_res)
                append_progenitor(tree, n_out, idx_first, m_prog1, level);
            if (m_prog2 > m_res)
                append_progenitor(tree, n_out, idx_first, m_prog2, level);
        }
        else {
            if (m_prog1 > m_res)
                tree_binary.push_back(Node(m_prog1, d_start + dd));
            if (m_prog2 > m_res)
                tree_binary.push_back(Node(m_prog2, d_start + dd));
        }
        //
        ++idx;
    }
    // Link progenitor indices
    if (n_out > 0) {
        sort(tree.begin() + idx_first, tree.end());
        tree[idx_root].idx_tail = idx_first;
        for(int i_prog = idx_first; i_prog - idx_first < n_out - 1; ++i_prog)
            tree[i_prog].idx_next = i_prog + 1;
    }
    return n_out;
}


MergerTree BinarySplit::grow(
    double m_root, double z_root, const vector<double>& z_target
)
{
    int level = 0;
    int idx_root = 0;
    int n_prog = 1;
    int n_prog_next = 1;
    double d_root = hr_->eval_d(z_root);
    double d_target;
    MergerTree tree;
    append_leaf(tree, m_root, level);
    for(auto z : z_target) {
        ++level;
        n_prog = n_prog_next;
        n_prog_next = 0;
        d_target = hr_->eval_d(z);
        for(int i_prog = 0; i_prog < n_prog; ++i_prog)
            n_prog_next += split(tree, idx_root++, level, d_root, d_target);
        d_root = d_target;
    }
    return tree;
}
