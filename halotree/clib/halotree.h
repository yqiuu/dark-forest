#ifndef PLANTER_H
#define PLANTER_H

#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<gsl/gsl_errno.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_spline.h>


typedef struct HaloRelation {
    gsl_interp_accel *sigma_acc;
    gsl_interp_accel *sigmasq_acc;
    gsl_interp_accel *alpha_acc;
    gsl_spline *sigma;
    gsl_spline *sigmasq;
    gsl_spline *alpha;

    gsl_interp_accel *Z2D_acc;
    gsl_interp_accel *D2Z_acc;
    gsl_spline *Z2D;
    gsl_spline *D2Z;
} HaloRelation;

void HaloRelation_Init(
    HaloRelation *hr, double *mass, double *sigma, double *sigmasq, double *alpha, int n_mass,
    double *z, double *delta_c, int n_z
);
void HaloRelation_Free(HaloRelation *hr);


typedef struct BinaryNode {
    double delta;
    double mass;
    int left;
    int right;
} BinaryNode;

typedef struct BinaryMergeTree {
    int n_node;
    int n_max;
    BinaryNode* node_arr;
    HaloRelation *hr;
} BinaryMergeTree;

int BinaryMergeTree_Nodes(
    BinaryMergeTree *tree, double z, double dz, BinaryNode *out, int n_max
);
void BinaryMergeTree_Init(
    BinaryMergeTree* tree, HaloRelation* hr, double m_root, double z_root, double z_target,
    double m_res, double eps1, double eps2, int n_base, gsl_rng* rstate
); 
void BinaryMergeTree_Free(BinaryMergeTree* tree);

#endif // PLANTER_H
