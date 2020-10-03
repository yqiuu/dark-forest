#ifndef NUT_H
#define NUT_H

#include<utility>
#include<algorithm>
#include<cmath>
#include<vector>
#include<functional>
#include<gsl/gsl_errno.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_spline.h>

namespace nut
{
class IntegralApproximation
{
public:
    IntegralApproximation(
        std::function<double(double)>, double a, double b,
        bool quadratic=true, double rtol=1e-6, double atol=1e-9,
        int depth_min=3
    );
    double integral() const {return integral_;}
    const std::vector<double>& x_knots() {return x_knots_;}
    const std::vector<double>& y_knots() {return y_knots_;}
    const std::vector<double>& s_knots() {return s_knots_;}
    double simps(double a, double b, double f_a, double f_m, double f_b)
        const {return (b - a)/6.*(f_a + 4.*f_m + f_b);}
    double trapz(double a, double b, double f_a, double f_b)
        const {return 0.5*(b - a)*(f_a + f_b);}

private:
    void init_params(double a, double b, std::vector<double>& params);
    void divide(std::vector<double>::iterator, std::vector<double>&, bool);
    void save_knots();
    // Parameters
    std::function<double(double)> func_;
    const bool quadratic_;
    const double rtol_;
    const double atol_;
    // Results
    double integral_;
    std::vector<double> x_knots_;
    std::vector<double> y_knots_;
    std::vector<double> s_knots_;
};


class InverseTranformSampler
{
public:
    InverseTranformSampler(IntegralApproximation proxy, gsl_rng* rstate);
    ~InverseTranformSampler()
    {
        gsl_spline_free(cdf_inv_);
        gsl_interp_accel_free(acc_);
    }
    double norm() const {return norm_;}
    double sample() const
        {return gsl_spline_eval(cdf_inv_, gsl_rng_uniform(rstate_), acc_);}
private:
    gsl_interp_accel* acc_;
    gsl_spline* cdf_inv_;
    gsl_rng* rstate_;
    double norm_;
};


class Selector
{
public:
    Selector(gsl_rng* rstate) : rstate_(rstate) {}
    size_t sample(std::vector<double>& weights);
private:
    gsl_rng* rstate_;
    std::vector<double> cdf_;
};


template<typename T>
void shuffle(std::vector<T>& target, gsl_rng* rstate)
{
    for (size_t idx = target.size() - 1; idx > 0; --idx)
        std::swap(target[idx], target[gsl_rng_uniform_int(rstate, idx)]);
}

} // dtree
#endif

