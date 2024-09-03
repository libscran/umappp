#include <gtest/gtest.h>

#include "umappp/find_ab.hpp"
#include <vector>

TEST(ABTest, FindAB) {
    // Comparing to values from uwot:::find_ab_params()
    auto out = umappp::internal::find_ab<double>(1, 0.05);
    EXPECT_LT(std::abs(out.first - 1.7502250), 1e-4);
    EXPECT_LT(std::abs(out.second - 0.8420554), 1e-4);

    out = umappp::internal::find_ab<double>(1, 0.01);
    EXPECT_LT(std::abs(out.first - 1.8956059), 1e-4);
    EXPECT_LT(std::abs(out.second - 0.800637), 1e-4);

    out = umappp::internal::find_ab<double>(2, 0.1);
    EXPECT_LT(std::abs(out.first - 0.5446606), 1e-4);
    EXPECT_LT(std::abs(out.second - 0.8420553), 1e-4);
}

TEST(ABTest, Dampening) {
    // Testing the dampening. This doesn't work by default with uwot, we need to fiddle with the parameters.
    // > spread <- 200
    // > min_dist <- 1
    // > xv <- seq(from = 0, to = spread * 3, length.out = 300)
    // > yv <- rep(0, length(xv))
    // > yv[xv < min_dist] <- 1
    // > yv[xv >= min_dist] <- exp(-(xv[xv >= min_dist] - min_dist)/spread)
    // > stats::nls(yv ~ 1/(1 + a * xv^(2 * b)), start = list(a = 0.000417953, b = 0.79548))$m$getPars()
    auto out = umappp::internal::find_ab<double>(200, 1);
    EXPECT_LT(std::abs(out.first - 0.0004176367), 1e-6);
    EXPECT_LT(std::abs(out.second - 0.7955526861), 1e-4);

    // Getting some code coverage for dampening failure.
    // This is achieved by forcing underflow of 'a'.
    out = umappp::internal::find_ab<double>(20, 1000);
    EXPECT_LT(out.first, 1e-100);
}
