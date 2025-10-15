#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <cmath>

TEST(ProbDistributionsGeneralizedGamma, reducesToLognormal) {
  const double y = 2.3;
  const double mu = 0.4;
  const double sigma = 1.1;
  const double Q = 0.0;

  EXPECT_FLOAT_EQ(stan::math::lognormal_lpdf(y, mu, sigma),
                  stan::math::generalized_gamma_lpdf(y, mu, sigma, Q));
  EXPECT_FLOAT_EQ(stan::math::lognormal_lcdf(y, mu, sigma),
                  stan::math::generalized_gamma_lcdf(y, mu, sigma, Q));
  EXPECT_FLOAT_EQ(stan::math::lognormal_lccdf(y, mu, sigma),
                  stan::math::generalized_gamma_lccdf(y, mu, sigma, Q));
}

TEST(ProbDistributionsGeneralizedGamma, reducesToWeibull) {
  const double y = 1.7;
  const double mu = 0.3;
  const double sigma = 0.8;
  const double Q = 1.0;

  const double weibull_shape = 1.0 / sigma;
  const double weibull_scale = std::exp(mu);

  EXPECT_FLOAT_EQ(stan::math::weibull_lpdf(y, weibull_shape, weibull_scale),
                  stan::math::generalized_gamma_lpdf(y, mu, sigma, Q));
  EXPECT_FLOAT_EQ(stan::math::weibull_lcdf(y, weibull_shape, weibull_scale),
                  stan::math::generalized_gamma_lcdf(y, mu, sigma, Q));
  EXPECT_FLOAT_EQ(stan::math::weibull_lccdf(y, weibull_shape, weibull_scale),
                  stan::math::generalized_gamma_lccdf(y, mu, sigma, Q));
}

TEST(ProbDistributionsGeneralizedGamma, rngSpecialCases) {
  const double mu = 0.2;
  const double sigma = 1.4;

  boost::random::mt19937 rng_ln(1234);
  boost::random::mt19937 rng_gen(1234);
  EXPECT_FLOAT_EQ(stan::math::lognormal_rng(mu, sigma, rng_ln),
                  stan::math::generalized_gamma_rng(mu, sigma, 0.0, rng_gen));

  boost::random::mt19937 rng(12345);
  EXPECT_NO_THROW(stan::math::generalized_gamma_rng(mu, sigma, 0.7, rng));
  EXPECT_NO_THROW(stan::math::generalized_gamma_rng(mu, sigma, -0.9, rng));
}
