#include"fstream"
#include"dtree.h"

using namespace std;
using namespace dtree;

inline void init_spline(
    gsl_interp_accel** spl_acc, gsl_spline** spl,
    double *x, double *y, int n_pts
)
{
    *spl_acc = gsl_interp_accel_alloc();
    *spl = gsl_spline_alloc(gsl_interp_cspline, n_pts);
    gsl_spline_init(*spl, x, y, n_pts);
}


void HaloRelation::init_object(
    double *mass, double *sigma, double *alpha, int n_mass,
    double *z, double *delta_c, int n_z
)
{
    init_spline(&sigma_acc_, &sigma_, mass, sigma, n_mass);
    init_spline(&alpha_acc_, &alpha_, mass, alpha, n_mass);
    init_spline(&Z2D_acc_, &Z2D_, z, delta_c, n_z);
    init_spline(&D2Z_acc_, &D2Z_, delta_c, z, n_z);
}


HaloRelation::HaloRelation(
    double *mass, double *sigma, double *alpha, int n_mass,
    double *z, double *delta_c, int n_z
)
{
    init_object(mass, sigma, alpha, n_mass, z, delta_c, n_z);
}


HaloRelation::HaloRelation(const char* fname)
{
    ifstream fp(fname, ios::in | ios::binary);
    if(!fp)
    {
        cout << "Error: cannot open the file" << endl;
        exit(EXIT_FAILURE);
    }

    int n_mass;
    fp.read((char*)&n_mass, sizeof(int));
    const size_t n_byte_m = n_mass*sizeof(double);
    double *mass = (double*)malloc(n_byte_m);
    fp.read((char*)mass, n_byte_m);
    double *sigma = (double*)malloc(n_byte_m);
    fp.read((char*)sigma, n_byte_m);
    double *alpha = (double*)malloc(n_byte_m);
    fp.read((char*)alpha, n_byte_m);

    int n_z;
    fp.read((char*)&n_z, sizeof(int));
    const size_t n_byte_z = n_z*sizeof(double);
    double *z = (double*)malloc(n_byte_z);
    fp.read((char*)z, n_byte_z);
    double *delta_c = (double*)malloc(n_byte_z);
    fp.read((char*)delta_c, n_byte_z);

    fp.close();

    init_object(mass, sigma, alpha, n_mass, z, delta_c, n_z);

    free(mass);
    free(sigma);
    free(alpha);
    free(z);
    free(delta_c);
}


HaloRelation::~HaloRelation()
{
    gsl_interp_accel_free(sigma_acc_);
    gsl_interp_accel_free(alpha_acc_);
    gsl_spline_free(sigma_);
    gsl_spline_free(alpha_);

    gsl_interp_accel_free(Z2D_acc_);
    gsl_interp_accel_free(D2Z_acc_);
    gsl_spline_free(Z2D_);
    gsl_spline_free(D2Z_);
}
