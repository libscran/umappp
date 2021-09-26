#include <gtest/gtest.h>

#include "umappp/find_ab.hpp"
#include <vector>

TEST(ABTest, FindAB) {
    // Comparing to values from uwot:::find_ab_params()
    auto out = umappp::find_ab<double>(1, 0.05);
    EXPECT_TRUE(std::abs(out.first - 1.7502250) < 1e-3);
    EXPECT_TRUE(std::abs(out.second - 0.8420554) < 1e-3);

    out = umappp::find_ab<double>(1, 0.01);
    EXPECT_TRUE(std::abs(out.first - 1.8956059) < 1e-3);
    EXPECT_TRUE(std::abs(out.second - 0.800637) < 1e-3);

    out = umappp::find_ab<double>(2, 0.1);
    EXPECT_TRUE(std::abs(out.first - 0.5446606) < 1e-3);
    EXPECT_TRUE(std::abs(out.second - 0.8420553) < 1e-3);
}
