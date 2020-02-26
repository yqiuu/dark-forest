#include"dtree.h"


HaloRelation HaloRelation_Read(char* fname) {
    FILE* fp = fopen(fname, "rb");
    if (fp == NULL) {
        printf("Error: cannot open the file\n");
        exit(EXIT_FAILURE);
    }

    int n_mass;
    fread(&n_mass, sizeof(int), 1, fp);
    double *mass = malloc(n_mass*sizeof(double));
    fread(mass, sizeof(double), n_mass, fp);
    double *sigma = malloc(n_mass*sizeof(double));
    fread(sigma, sizeof(double), n_mass, fp);
    double *alpha = malloc(n_mass*sizeof(double));
    fread(alpha, sizeof(double), n_mass, fp);

    int n_z;
    fread(&n_z, sizeof(int), 1, fp);
    double *z = malloc(n_z*sizeof(double));
    fread(z, sizeof(double), n_z, fp);
    double *delta_c = malloc(n_z*sizeof(double));
    fread(delta_c, sizeof(double), n_z, fp);

    fclose(fp);

    return HaloRelation_Init(mass, sigma, alpha, n_mass, z, delta_c, n_z);
}


HaloRelation HaloRelation_Init(
    double *mass, double *sigma, double *alpha, int n_mass, double *z, double *delta_c, int n_z
) {
    HaloRelation hr;
    hr.sigma_acc = gsl_interp_accel_alloc();
    hr.sigma = gsl_spline_alloc(gsl_interp_cspline, n_mass);
    gsl_spline_init(hr.sigma, mass, sigma, n_mass);

    hr.alpha_acc = gsl_interp_accel_alloc();
    hr.alpha = gsl_spline_alloc(gsl_interp_cspline, n_mass);
    gsl_spline_init(hr.alpha, mass, alpha, n_mass);

    hr.Z2D_acc = gsl_interp_accel_alloc();
    hr.Z2D = gsl_spline_alloc(gsl_interp_cspline, n_z);
    gsl_spline_init(hr.Z2D, z, delta_c, n_z);

    hr.D2Z_acc = gsl_interp_accel_alloc();
    hr.D2Z = gsl_spline_alloc(gsl_interp_cspline, n_z);
    gsl_spline_init(hr.D2Z, delta_c, z, n_z);

    return hr;
}


void HaloRelation_Free(HaloRelation *hr) {
    gsl_spline_free(hr->sigma);
    gsl_spline_free(hr->alpha);
    gsl_interp_accel_free(hr->sigma_acc);
    gsl_interp_accel_free(hr->alpha_acc);

    gsl_spline_free(hr->Z2D);
    gsl_spline_free(hr->D2Z);
    gsl_interp_accel_free(hr->Z2D_acc);
    gsl_interp_accel_free(hr->D2Z_acc);
}
