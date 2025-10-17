#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

TEST(ProbDistributionsLoglogistic, lcdfMatchesCdf) {
  const double y = 2.5;
  const double alpha = 1.7;
  const double beta = 2.3;

  const double expected = std::log(stan::math::loglogistic_cdf(y, alpha, beta));
  EXPECT_FLOAT_EQ(expected, stan::math::loglogistic_lcdf(y, alpha, beta));
}

TEST(ProbDistributionsLoglogistic, lccdfMatchesComplement) {
  const double y = 2.5;
  const double alpha = 1.7;
  const double beta = 2.3;

  const double expected
      = std::log1p(-stan::math::loglogistic_cdf(y, alpha, beta));
  EXPECT_FLOAT_EQ(expected, stan::math::loglogistic_lccdf(y, alpha, beta));
}

TEST(ProbDistributionsLoglogistic, lcdfVectorSum) {
  std::vector<double> y{1.2, 3.0};
  const double alpha = 2.1;
  const double beta = 1.5;

  double expected = 0.0;
  for (const auto& yi : y) {
    expected += std::log(stan::math::loglogistic_cdf(yi, alpha, beta));
  }
  EXPECT_FLOAT_EQ(expected, stan::math::loglogistic_lcdf(y, alpha, beta));
}

TEST(ProbDistributionsLoglogistic, lccdfVectorSum) {
  std::vector<double> y{1.2, 3.0};
  const double alpha = 2.1;
  const double beta = 1.5;

  double expected = 0.0;
  for (const auto& yi : y) {
    expected += std::log1p(-stan::math::loglogistic_cdf(yi, alpha, beta));
  }
  EXPECT_FLOAT_EQ(expected, stan::math::loglogistic_lccdf(y, alpha, beta));
}

TEST(ProbDistributionsLoglogistic, lcdfZero) {
  const double alpha = 1.5;
  const double beta = 2.0;
  EXPECT_EQ(stan::math::NEGATIVE_INFTY,
            stan::math::loglogistic_lcdf(0.0, alpha, beta));
}

TEST(ProbDistributionsLoglogistic, lccdfZero) {
  const double alpha = 1.5;
  const double beta = 2.0;
  EXPECT_DOUBLE_EQ(0.0, stan::math::loglogistic_lccdf(0.0, alpha, beta));
}
