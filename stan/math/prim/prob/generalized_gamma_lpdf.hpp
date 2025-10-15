#ifndef STAN_MATH_PRIM_PROB_GENERALIZED_GAMMA_LPDF_HPP
#define STAN_MATH_PRIM_PROB_GENERALIZED_GAMMA_LPDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/fun/as_array_or_scalar.hpp>
#include <stan/math/prim/fun/as_value_column_array_or_scalar.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/meta/include_summand.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/promote_scalar.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Returns the log PDF of the generalized gamma distribution as defined by
 * Prentice (1974). Given matching container sizes, returns the sum of the log
 * densities.
 *
 * @tparam T_y type of random variable
 * @tparam T_loc type of location parameter
 * @tparam T_scale type of scale parameter
 * @tparam T_shape type of shape parameter
 * @param y random variable (non-negative)
 * @param mu location parameter (real)
 * @param sigma scale parameter (positive)
 * @param Q shape parameter (real)
 */
template <bool propto, typename T_y, typename T_loc, typename T_scale,
          typename T_shape,
          require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_y, T_loc, T_scale, T_shape>* = nullptr>
return_type_t<T_y, T_loc, T_scale, T_shape> generalized_gamma_lpdf(
    const T_y& y, const T_loc& mu, const T_scale& sigma, const T_shape& Q) {
  using T_partials_return =
      partials_return_t<T_y, T_loc, T_scale, T_shape>;
  using T_y_ref = ref_type_if_not_constant_t<T_y>;
  using T_mu_ref = ref_type_if_not_constant_t<T_loc>;
  using T_sigma_ref = ref_type_if_not_constant_t<T_scale>;
  using T_Q_ref = ref_type_if_not_constant_t<T_shape>;
  static constexpr const char* function = "generalized_gamma_lpdf";

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
  if constexpr (!include_summand<propto, T_y, T_loc, T_scale, T_shape>::value) {
    return 0.0;
  }

  auto ops_partials = make_partials_propagator(y_ref, mu_ref, sigma_ref, Q_ref);

  if (sum(promote_scalar<int>(y_val == 0))) {
    return ops_partials.build(LOG_ZERO);
  }

  scalar_seq_view<T_y_ref> y_vec(y_ref);
  scalar_seq_view<T_mu_ref> mu_vec(mu_ref);
  scalar_seq_view<T_sigma_ref> sigma_vec(sigma_ref);
  scalar_seq_view<T_Q_ref> Q_vec(Q_ref);

  size_t N = max_size(y, mu, sigma, Q);
  T_partials_return logp = 0.0;

  for (size_t n = 0; n < N; ++n) {
    const auto y_dbl = value_of(y_vec.val(n));
    const auto mu_dbl = value_of(mu_vec.val(n));
    const auto sigma_dbl = value_of(sigma_vec.val(n));
    const auto Q_dbl = value_of(Q_vec.val(n));

    const double log_y = std::log(y_dbl);
    const double inv_sigma = 1.0 / sigma_dbl;
    const double logy_minus_mu = log_y - mu_dbl;

    if (Q_dbl == 0.0) {
      const double inv_sigma_sq = inv_sigma * inv_sigma;
      if constexpr (include_summand<propto>::value) {
        logp += NEG_LOG_SQRT_TWO_PI;
      }
      if constexpr (include_summand<propto, T_scale>::value) {
        logp -= std::log(sigma_dbl);
      }
      if constexpr (include_summand<propto, T_y>::value) {
        logp -= log_y;
      }
      const double logy_m_mu_div_sigma_sq = logy_minus_mu * inv_sigma_sq;
      logp -= 0.5 * logy_minus_mu * logy_m_mu_div_sigma_sq;

      if constexpr (is_autodiff_v<T_y>) {
        partials<0>(ops_partials)[n]
            += -(1.0 + logy_m_mu_div_sigma_sq) / y_dbl;
      }
      if constexpr (is_autodiff_v<T_loc>) {
        partials<1>(ops_partials)[n] += logy_m_mu_div_sigma_sq;
      }
      if constexpr (is_autodiff_v<T_scale>) {
        partials<2>(ops_partials)[n]
            += (logy_m_mu_div_sigma_sq * logy_minus_mu - 1.0) * inv_sigma;
      }
      if constexpr (is_autodiff_v<T_shape>) {
        const double sigma_cubed = sigma_dbl * sigma_dbl * sigma_dbl;
        partials<3>(ops_partials)[n]
            += -(logy_minus_mu * logy_minus_mu * logy_minus_mu)
               / (6.0 * sigma_cubed);
      }
      continue;
    }

    const double abs_Q = std::abs(Q_dbl);
    const double log_abs_Q = std::log(abs_Q);
    const double inv_sq = 1.0 / (Q_dbl * Q_dbl);
    const double w = logy_minus_mu * inv_sigma;
    const double Qw = Q_dbl * w;
    const double exp_Qw = std::exp(Qw);

    if constexpr (include_summand<propto, T_scale>::value) {
      logp -= std::log(sigma_dbl);
    }
    if constexpr (include_summand<propto, T_y>::value) {
      logp -= log_y;
    }
    logp += log_abs_Q - lgamma(inv_sq)
            + inv_sq * (Qw - exp_Qw - 2.0 * log_abs_Q);

    if constexpr (is_autodiff_v<T_y>) {
      partials<0>(ops_partials)[n]
          += -1.0 / y_dbl + (1.0 - exp_Qw) / (Q_dbl * sigma_dbl * y_dbl);
    }
    if constexpr (is_autodiff_v<T_loc>) {
      partials<1>(ops_partials)[n] += (exp_Qw - 1.0) / (Q_dbl * sigma_dbl);
    }
    if constexpr (is_autodiff_v<T_scale>) {
      partials<2>(ops_partials)[n]
          += (-1.0 + (exp_Qw - 1.0) * w / Q_dbl) * inv_sigma;
    }
    if constexpr (is_autodiff_v<T_shape>) {
      const double digamma_val = digamma(inv_sq);
      partials<3>(ops_partials)[n]
          += 1.0 / Q_dbl
             + (2.0 * inv_sq * digamma_val
                - 2.0 * inv_sq * (Qw - exp_Qw - 2.0 * log_abs_Q + 1.0))
                   / Q_dbl
             + inv_sq * w * (1.0 - exp_Qw);
    }
  }

  return ops_partials.build(logp);
}

template <typename T_y, typename T_loc, typename T_scale, typename T_shape>
inline return_type_t<T_y, T_loc, T_scale, T_shape> generalized_gamma_lpdf(
    const T_y& y, const T_loc& mu, const T_scale& sigma, const T_shape& Q) {
  return generalized_gamma_lpdf<false>(y, mu, sigma, Q);
}

}  // namespace math
}  // namespace stan

#endif
