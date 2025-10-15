#ifndef STAN_MATH_PRIM_PROB_GENERALIZED_GAMMA_RNG_HPP
#define STAN_MATH_PRIM_PROB_GENERALIZED_GAMMA_RNG_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/prob/gamma_rng.hpp>
#include <stan/math/prim/prob/lognormal_rng.hpp>
#include <stan/math/prim/meta/VectorBuilder.hpp>
#include <cmath>

namespace stan {
namespace math {

template <typename T_loc, typename T_scale, typename T_shape, class RNG>
inline typename VectorBuilder<true, double, T_loc, T_scale, T_shape>::type
    generalized_gamma_rng(const T_loc& mu, const T_scale& sigma,
                          const T_shape& Q, RNG& rng) {
  using T_mu_ref = ref_type_t<T_loc>;
  using T_sigma_ref = ref_type_t<T_scale>;
  using T_Q_ref = ref_type_t<T_shape>;
  static constexpr const char* function = "generalized_gamma_rng";

  check_consistent_sizes(function, "Location parameter", mu,
                         "Scale parameter", sigma, "Shape parameter", Q);
  T_mu_ref mu_ref = mu;
  T_sigma_ref sigma_ref = sigma;
  T_Q_ref Q_ref = Q;

  check_finite(function, "Location parameter", mu_ref);
  check_positive_finite(function, "Scale parameter", sigma_ref);
  check_finite(function, "Shape parameter", Q_ref);

  scalar_seq_view<T_mu_ref> mu_vec(mu_ref);
  scalar_seq_view<T_sigma_ref> sigma_vec(sigma_ref);
  scalar_seq_view<T_Q_ref> Q_vec(Q_ref);
  size_t N = max_size(mu, sigma, Q);
  VectorBuilder<true, double, T_loc, T_scale, T_shape> output(N);

  for (size_t n = 0; n < N; ++n) {
    const double mu_dbl = value_of(mu_vec.val(n));
    const double sigma_dbl = value_of(sigma_vec.val(n));
    const double Q_dbl = value_of(Q_vec.val(n));

    if (Q_dbl == 0.0) {
      output[n] = lognormal_rng(mu_dbl, sigma_dbl, rng);
    } else {
      const double inv_sq = 1.0 / (Q_dbl * Q_dbl);
      const double gamma_sample = gamma_rng(inv_sq, 1.0, rng);
      const double w = std::log(Q_dbl * Q_dbl * gamma_sample) / Q_dbl;
      output[n] = std::exp(mu_dbl + sigma_dbl * w);
    }
  }

  return output.data();
}

}  // namespace math
}  // namespace stan

#endif
