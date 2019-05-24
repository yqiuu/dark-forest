#ifndef HALOTREE_H
#define HALOTREE_H

#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<gsl/gsl_errno.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_spline.h>


typedef struct SplitKit {
    double G0;
    double gamma1;
    double gamma2;
    double m_res;
    double eps1;
    double eps2;

    double J_bp1;
    double J_bp2;
    gsl_interp_accel *J_u_acc;
    gsl_spline *J_u;
} SplitKit;

double SplitKit_J(SplitKit *kit_sp, double u_res);
SplitKit SplitKit_Init(
    double G0, double gamma1, double gamma2, double m_res, double eps1, double eps2
);
void SplitKit_Free(SplitKit *kit_sp);


typedef struct AugmentKit {
    int n_trail;
    int n_max_trail;
    double eps;
    double m_cut;
} AugmentKit;
AugmentKit AugmentKit_Init(int n_trail, int n_max_trail, double eps, double m_cut);


typedef struct HaloRelation {
    gsl_interp_accel *sigma_acc;
    gsl_interp_accel *alpha_acc;
    gsl_spline *sigma;
    gsl_spline *alpha;

    gsl_interp_accel *Z2D_acc;
    gsl_interp_accel *D2Z_acc;
    gsl_spline *Z2D;
    gsl_spline *D2Z;
} HaloRelation;

HaloRelation HaloRelation_Read(char* fname);
HaloRelation HaloRelation_Init(
    double *mass, double *sigma, double *alpha, int n_mass, double *z, double *delta_c, int n_z
);
void HaloRelation_Free(HaloRelation *hr);


typedef struct Node {
    double trace;
    double mass;
    int idx_tail;
    int idx_next;
} Node;

typedef struct NodeVector {
    int n_max;
    int n_node;
    Node* data;
} NodeVector;

int NodeVector_Append(NodeVector *vector, double delta, double mass);
NodeVector NodeVector_Init(int n_base);
void NodeVector_Free(NodeVector *vector);


NodeVector MergeTree_Grow(
    double m_root, double z_root, double *z_target, int n_target,
    HaloRelation* hr, SplitKit* kit_sp, gsl_rng* rstate
);

NodeVector MergeTree_Augment(
    double *m_prog, int n_prog, double m_root, double z_root, double *z_target, int n_target,
    HaloRelation* hr, AugmentKit *kit_ag, SplitKit *kit_sp, gsl_rng* rstate, int n_base
);
#endif // HALOTREE_H
