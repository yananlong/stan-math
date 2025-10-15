#ifndef STAN_MATH_PRIM_PROB_LOGLOGISTIC_LCCDF_HPP
#define STAN_MATH_PRIM_PROB_LOGLOGISTIC_LCCDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/fun/as_array_or_scalar.hpp>
#include <stan/math/prim/fun/as_value_column_array_or_scalar.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/log1p.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/pow.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/promote_scalar.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Returns the log complementary cumulative distribution function for the
 * loglogistic distribution.
 *
 * @tparam T_y type of random variable
 * @tparam T_scale type of scale parameter
 * @tparam T_shape type of shape parameter
 * @param y random variable
 * @param alpha scale parameter
 * @param beta shape parameter
 * @return log complementary CDF evaluated at the inputs
 */
template <typename T_y, typename T_scale, typename T_shape,
          require_all_not_nonscalar_prim_or_rev_kernel_expression_t<
              T_y, T_scale, T_shape>* = nullptr>
return_type_t<T_y, T_scale, T_shape> loglogistic_lccdf(const T_y& y,
                                                       const T_scale& alpha,
                                                       const T_shape& beta) {
  using T_partials_return = partials_return_t<T_y, T_scale, T_shape>;
  using T_y_ref = ref_type_if_not_constant_t<T_y>;
  using T_alpha_ref = ref_type_if_not_constant_t<T_scale>;
  using T_beta_ref = ref_type_if_not_constant_t<T_shape>;
  static constexpr const char* function = "loglogistic_lccdf";

  T_y_ref y_ref = y;
  T_alpha_ref alpha_ref = alpha;
  T_beta_ref beta_ref = beta;

  decltype(auto) y_val = to_ref(as_value_column_array_or_scalar(y_ref));
  decltype(auto) alpha_val = to_ref(as_value_column_array_or_scalar(alpha_ref));
  decltype(auto) beta_val = to_ref(as_value_column_array_or_scalar(beta_ref));

  check_nonnegative(function, "Random variable", y_val);
  check_positive_finite(function, "Scale parameter", alpha_val);
  check_positive_finite(function, "Shape parameter", beta_val);

  if (size_zero(y, alpha, beta)) {
    return 0.0;
  }

  auto ops_partials = make_partials_propagator(y_ref, alpha_ref, beta_ref);

  if (sum(promote_scalar<int>(y_val == 0))) {
    return ops_partials.build(0.0);
  }

  const auto& alpha_div_y = to_ref(alpha_val / y_val);
  const auto& pow_term = to_ref(pow(alpha_div_y, beta_val));
  const auto& log_term = to_ref(log(alpha_div_y));
  const auto& log_pow = to_ref(beta_val * log_term);
  const auto& log1p_term = to_ref(log1p(pow_term));

  T_partials_return log_ccdf = sum(log_pow - log1p_term);

  if constexpr (is_any_autodiff_v<T_y, T_scale, T_shape>) {
    const auto& inv_one_plus_pow
        = to_ref_if<is_autodiff_v<T_y> + is_autodiff_v<T_scale>
                    + is_autodiff_v<T_shape> >= 2>(1.0 / (1.0 + pow_term));
    if constexpr (is_autodiff_v<T_y>) {
      partials<0>(ops_partials) = -beta_val * inv_one_plus_pow / y_val;
    }
    if constexpr (is_autodiff_v<T_scale>) {
      partials<1>(ops_partials) = beta_val * inv_one_plus_pow / alpha_val;
    }
    if constexpr (is_autodiff_v<T_shape>) {
      partials<2>(ops_partials) = inv_one_plus_pow * log_term;
    }
  }

  return ops_partials.build(log_ccdf);
}

}  // namespace math
}  // namespace stan

#endif
