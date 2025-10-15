#ifndef STAN_MATH_PRIM_PROB_GENERALIZED_GAMMA_LCDF_HPP
#define STAN_MATH_PRIM_PROB_GENERALIZED_GAMMA_LCDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/fun/as_array_or_scalar.hpp>
#include <stan/math/prim/fun/as_value_column_array_or_scalar.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/erfc.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/gamma_p.hpp>
#include <stan/math/prim/fun/gamma_q.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/fun/grad_reg_inc_gamma.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/tgamma.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/promote_scalar.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <cmath>
#include <limits>

namespace stan {
namespace math {

template <typename T_y, typename T_loc, typename T_scale, typename T_shape,
          require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_y, T_loc, T_scale, T_shape>* = nullptr>
return_type_t<T_y, T_loc, T_scale, T_shape> generalized_gamma_lcdf(
    const T_y& y, const T_loc& mu, const T_scale& sigma, const T_shape& Q) {
  using T_partials_return =
      partials_return_t<T_y, T_loc, T_scale, T_shape>;
  using T_y_ref = ref_type_if_not_constant_t<T_y>;
  using T_mu_ref = ref_type_if_not_constant_t<T_loc>;
  using T_sigma_ref = ref_type_if_not_constant_t<T_scale>;
  using T_Q_ref = ref_type_if_not_constant_t<T_shape>;
  static constexpr const char* function = "generalized_gamma_lcdf";

  T_y_ref y_ref = y;
  T_mu_ref mu_ref = mu;
  T_sigma_ref sigma_ref = sigma;
  T_Q_ref Q_ref = Q;

  decltype(auto) y_val = to_ref(as_value_column_array_or_scalar(y_ref));
  decltype(auto) mu_val = to_ref(as_value_column_array_or_scalar(mu_ref));
  decltype(auto) sigma_val = to_ref(as_value_column_array_or_scalar(sigma_ref));
  decltype(auto) Q_val = to_ref(as_value_column_array_or_scalar(Q_ref));

  check_nonnegative(function, "Random variable", y_val);
  check_finite(function, "Location parameter", mu_val);
  check_positive_finite(function, "Scale parameter", sigma_val);
  check_finite(function, "Shape parameter", Q_val);

  if (size_zero(y, mu, sigma, Q)) {
    return 0.0;
  }

  auto ops_partials = make_partials_propagator(y_ref, mu_ref, sigma_ref, Q_ref);

  if (sum(promote_scalar<int>(y_val == 0))) {
    return ops_partials.build(NEGATIVE_INFTY);
  }

  scalar_seq_view<T_y_ref> y_vec(y_ref);
  scalar_seq_view<T_mu_ref> mu_vec(mu_ref);
  scalar_seq_view<T_sigma_ref> sigma_vec(sigma_ref);
  scalar_seq_view<T_Q_ref> Q_vec(Q_ref);

  size_t N = max_size(y, mu, sigma, Q);
  T_partials_return log_cdf = 0.0;

  for (size_t n = 0; n < N; ++n) {
    const auto y_dbl = value_of(y_vec.val(n));
    const auto mu_dbl = value_of(mu_vec.val(n));
    const auto sigma_dbl = value_of(sigma_vec.val(n));
    const auto Q_dbl = value_of(Q_vec.val(n));

    const double log_y = std::log(y_dbl);
    const double inv_sigma = 1.0 / sigma_dbl;
    const double logy_minus_mu = log_y - mu_dbl;

    if (Q_dbl == 0.0) {
      const double scaled_diff =
          (logy_minus_mu) / (sigma_dbl * SQRT_TWO);
      const double erfc_val = erfc(-scaled_diff);
      log_cdf += LOG_HALF + std::log(erfc_val);

      if constexpr (is_any_autodiff_v<T_y, T_loc, T_scale, T_shape>) {
        const double rep_deriv =
            -SQRT_TWO_OVER_SQRT_PI * std::exp(-scaled_diff * scaled_diff)
            / (sigma_dbl * erfc_val);
        if constexpr (is_autodiff_v<T_y>) {
          partials<0>(ops_partials)[n] += -rep_deriv / y_dbl;
        }
        if constexpr (is_autodiff_v<T_loc>) {
          partials<1>(ops_partials)[n] += rep_deriv;
        }
        if constexpr (is_autodiff_v<T_scale>) {
          partials<2>(ops_partials)[n]
              += rep_deriv * scaled_diff * SQRT_TWO;
        }
        if constexpr (is_autodiff_v<T_shape>) {
          const double eps
              = std::sqrt(std::numeric_limits<double>::epsilon());
          auto logcdf_eval = [&](double q_val) {
            const double inv_sq = 1.0 / (q_val * q_val);
            const double w = logy_minus_mu * inv_sigma;
            const double Qw = q_val * w;
            const double exp_Qw = std::exp(Qw);
            const double z = exp_Qw / (q_val * q_val);
            if (q_val > 0) {
              return std::log(gamma_p(inv_sq, z));
            } else {
              return std::log(gamma_q(inv_sq, z));
            }
          };
          const double deriv
              = (logcdf_eval(eps) - logcdf_eval(-eps)) / (2.0 * eps);
          partials<3>(ops_partials)[n] += deriv;
        }
      }
      continue;
    }

    const double abs_Q = std::abs(Q_dbl);
    const double log_abs_Q = std::log(abs_Q);
    const double inv_sq = 1.0 / (Q_dbl * Q_dbl);
    const double w = logy_minus_mu * inv_sigma;
    const double Qw = Q_dbl * w;
    const double exp_Qw = std::exp(Qw);
    const double z = exp_Qw / (Q_dbl * Q_dbl);
    const double sign_Q = Q_dbl > 0.0 ? 1.0 : -1.0;

    double cdf_val = 0.0;
    if (Q_dbl > 0.0) {
      cdf_val = gamma_p(inv_sq, z);
    } else {
      cdf_val = gamma_q(inv_sq, z);
    }

    log_cdf += std::log(cdf_val);

    if constexpr (is_any_autodiff_v<T_y, T_loc, T_scale, T_shape>) {
      const double log_pdf =
          log_abs_Q - std::log(sigma_dbl) - log_y - lgamma(inv_sq)
          + inv_sq * (Qw - exp_Qw - 2.0 * log_abs_Q);
      const double pdf = std::exp(log_pdf);
      const double dz_dy = exp_Qw / (Q_dbl * y_dbl * sigma_dbl);
      const double gamma_pdf = pdf / (sign_Q * dz_dy);
      const double dz_dmu = -exp_Qw / (Q_dbl * sigma_dbl);
      const double dz_dsigma = -(exp_Qw / Q_dbl) * w / sigma_dbl;
      const double dz_dQ = exp_Qw * (w / (Q_dbl * Q_dbl) - 2.0 / (Q_dbl * Q_dbl * Q_dbl));
      const double dk_dQ = -2.0 / (Q_dbl * Q_dbl * Q_dbl);
      const double gamma_val = tgamma(inv_sq);
      const double digamma_val = digamma(inv_sq);
      const double grad_reg =
          grad_reg_inc_gamma(inv_sq, z, gamma_val, digamma_val);

      if constexpr (is_autodiff_v<T_y>) {
        partials<0>(ops_partials)[n] += pdf / cdf_val;
      }
      if constexpr (is_autodiff_v<T_loc>) {
        const double dcdf = sign_Q * gamma_pdf * dz_dmu;
        partials<1>(ops_partials)[n] += dcdf / cdf_val;
      }
      if constexpr (is_autodiff_v<T_scale>) {
        const double dcdf = sign_Q * gamma_pdf * dz_dsigma;
        partials<2>(ops_partials)[n] += dcdf / cdf_val;
      }
      if constexpr (is_autodiff_v<T_shape>) {
        const double dcdf = sign_Q * (gamma_pdf * dz_dQ + grad_reg * dk_dQ);
        partials<3>(ops_partials)[n] += dcdf / cdf_val;
      }
    }
  }

  return ops_partials.build(log_cdf);
}

}  // namespace math
}  // namespace stan

#endif
