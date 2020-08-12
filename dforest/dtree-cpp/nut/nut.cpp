#include"nut.h"

using namespace std;
using namespace nut;

inline bool isclose(double a, double b, double rtol, double atol)
{
    return abs(a - b) <= max(rtol*max(abs(a), abs(b)), atol);
}


IntegralApproximation::IntegralApproximation(
    function<double(double)> func, double a, double b,
    bool quadratic, double rtol, double atol, int depth_min
) : func_(func), quadratic_(quadratic), rtol_(rtol), atol_(atol)
{
    const int n_param = quadratic ? 7 : 5;
    vector<double> params;
    vector<double> params_next;

    init_params(a, b, params);
    for (int i_depth = 0; params.size() != 0; ++i_depth)
    {
        for(auto it = params.begin(); it != params.end(); it += n_param)
            divide(it, params_next, i_depth >= depth_min);
        params.clear();
        params.swap(params_next);
    }

    save_knots();
}


void IntegralApproximation::init_params(
    double a, double b, vector<double>& params
)
{
    double m = 0.5*(a + b);
    double f_a = func_(a);
    double f_b = func_(b);
    double f_m;
    double s_ab;
    if (quadratic_) {
        f_m = func_(m);
        s_ab = simps(a, b, f_a, f_m, f_b);
        params.insert(params.end(), {a, f_a, m, f_m, b, f_b, s_ab});
    }
    else {
        s_ab = trapz(a, b, f_a, f_b);
        params.insert(params.end(), {a, f_a, b, f_b, s_ab});
    } 
    x_knots_.push_back(a);
    y_knots_.push_back(f_a);
    s_knots_.push_back(0.);
}


void IntegralApproximation::divide(
    vector<double>::iterator it, vector<double>& params_next, bool depth_limit
)
{
    double a, m, b, f_a, f_m, f_b, s_ab;
    double ml, f_ml, s_am, mr, f_mr, s_mb, s_amb;
    if (quadratic_)
    {
        a = *it++;
        f_a = *it++;
        m = *it++;
        f_m = *it++;
        b = *it++;
        f_b = *it++;
        s_ab = *it++;
        // Compute the left integral
        ml = 0.5*(a + m);
        f_ml = func_(ml);
        s_am = simps(a, m, f_a, f_ml, f_m);
        // Compute right integral
        mr = 0.5*(m + b);
        f_mr = func_(mr);
        s_mb = simps(m, b, f_m, f_mr, f_b);
    }
    else
    {
        a = *it++;
        f_a = *it++;
        b = *it++;
        f_b = *it++;
        s_ab = *it++;
        // Compute left and right integrals
        m = 0.5*(a + b);
        f_m = func_(m);
        s_am = trapz(a, m, f_a, f_m);
        s_mb = trapz(m, b, f_m, f_b);
    }
    //
    int disp;
    s_amb = s_am + s_mb;
    if (isclose(s_ab, s_amb, rtol_, atol_) && depth_limit)
    {
        disp = lower_bound(x_knots_.begin(), x_knots_.end(), b) \
            - x_knots_.begin();
        x_knots_.insert(x_knots_.begin() + disp, b);
        y_knots_.insert(y_knots_.begin() + disp, f_b);
        s_knots_.insert(s_knots_.begin() + disp, s_amb);
    }
    else
    {
        if (quadratic_)
        {
            params_next.insert(
                params_next.end(), {a, f_a, ml, f_ml, m, f_m, s_am}
            );
            params_next.insert(
                params_next.end(), {m ,f_m, mr, f_mr, b, f_b, s_mb}
            );
        }
        else
        {
            params_next.insert(params_next.end(), {a, f_a, m, f_m, s_am});
            params_next.insert(params_next.end(), {m, f_m, b, f_b, s_mb});
        }
    }
}


void IntegralApproximation::save_knots()
{
    size_t n_knot = s_knots_.size();
    vector<double> s_knots(n_knot);
    // Compute cumulative sum
    s_knots[0] = s_knots_[0];
    for(size_t ii = 1; ii != n_knot; ++ii)
        s_knots[ii]  = s_knots[ii - 1] + s_knots_[ii];
    // Normalise
    integral_ = s_knots[n_knot - 1];
    s_knots_ = move(s_knots);
}


InverseTranformSampler::InverseTranformSampler(
    function<double(double)> pdf, double a, double b, gsl_rng* rstate,
    bool quadratic, double rtol, double atol
) : rstate_(rstate)
{
    IntegralApproximation proxy(pdf, a, b, quadratic, rtol, atol);
    // Set normalisation
    norm_ = proxy.integral();
    // Remove duplicated values
    int n_knot = proxy.x_knots().size();
    double val = proxy.s_knots()[0];
    int idx_begin = 0;
    for(int i_knot = 1; i_knot < n_knot; ++i_knot)
    {
        if (proxy.s_knots()[i_knot] == val)
            ++idx_begin;
        else
            break;
    }
    val = proxy.s_knots()[n_knot - 1];
    int idx_end = n_knot;
    for(int i_knot = n_knot - 2; i_knot > idx_begin; --i_knot)
    {
        if (proxy.s_knots()[i_knot] == val)
            --idx_end;
        else
            break;
    }
    int idx = 0;
    int n_interp = idx_end - idx_begin;
    double* x_knots = (double*)malloc(n_interp*sizeof(double));
    double* s_knots = (double*)malloc(n_interp*sizeof(double));
    for(int i_knot = idx_begin; i_knot < idx_end; ++i_knot)
    {
        x_knots[idx] = proxy.x_knots()[i_knot];
        s_knots[idx] = proxy.s_knots()[i_knot]/norm_;
        ++idx;
    }
    // Construct CDF
    acc_ =  gsl_interp_accel_alloc();
    cdf_inv_ = gsl_spline_alloc(gsl_interp_steffen, n_interp);
    gsl_spline_init(cdf_inv_, s_knots, x_knots, n_interp);
    //
    free(x_knots);
    free(s_knots);
}

