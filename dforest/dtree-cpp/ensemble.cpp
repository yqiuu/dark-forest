#include"dtree.h"
#include<iostream>

#define TOL_RES 1e-6

using namespace std;
using namespace dtree;
using namespace nut;

void EnsembleGrower::sample(double m2, double z1, double z2)
{
    prog_func_.update(m2, z1, z2);
    IntegralApproximation proxy(prog_func_, m_res_, m2, true, 1e-6, 1e-9, 3);
    if (m2/m_res_ - 1 < TOL_RES || proxy.integral() == 0.)
        return;
    InverseTranformSampler sampler(proxy, rstate_);
    //
    int n_prog = (int)(sampler.norm()*n_draw_);
    vector<double> m_prog(n_prog);
    for (auto& m : m_prog)
        m = sampler.sample();
    sort(m_prog.begin(), m_prog.end(), greater<double>());
    //
    for (auto& elem : pool_prog_)
        elem.clear();
    //
    vector<double> m_free(n_draw_);
    vector<double> weights(n_draw_);
    Selector selector(rstate_);
    bool reject;
    int idx;
    for (auto& m : m_free)
        m = m2;
    for (auto m : m_prog)
    {
        weights = m_free;
        for (auto& w : weights)
            if (w <= m)
                w = 0.;
        reject = true;
        for (auto w : weights)
            if (w != 0.)
            {
                reject = false;
                break;
            }
        if (reject)
            continue;
        idx = selector.sample(weights);
        pool_prog_[idx].push_back(m); 
        m_free[idx] -= m;
    }
    shuffle(pool_prog_, rstate_);
}


int EnsembleGrower::grow_level(
    MergerTree& tree, int idx_root, int level, double z_target, double z_root
)
{
    double m_root = tree[idx_root].mass;
    if (abs(m_prev_ - m_root) >= m_eps_ || i_draw_ == n_draw_)
    {
        sample(m_root, z_target, z_root);
        i_draw_ = 0;
    }
    m_prev_ = m_root;
    //
    auto it = pool_prog_[i_draw_].begin();
    auto it_end = pool_prog_[i_draw_].end() - 1;
    int idx_next = tree.size();
    tree[idx_root].idx_tail = idx_next;
    for (; it < it_end; ++it)
        tree.push_back(Leaf(*it, level, -1, idx_next++));
    tree.push_back(Leaf(*it, level, -1, -1));
    return pool_prog_[i_draw_++].size();
}


MergerTree EnsembleGrower::grow_tree(
    double m_root, double z_root, const vector<double>& z_target
)
{
    int level = 0;
    int idx_root = 0;
    int n_prog = 1;
    int n_prog_next = 1;
    MergerTree tree;
    tree.push_back(Leaf(m_root, level, -1, -1));
    for (auto z : z_target)
    {
        ++level;
        n_prog = n_prog_next;
        n_prog_next = 0;
        for (int i_prog = 0; i_prog < n_prog; ++i_prog)
            n_prog_next += \
                grow_level(tree, idx_root++, level, z_root, z);
        z_root = z;
    }
    return tree;
}
