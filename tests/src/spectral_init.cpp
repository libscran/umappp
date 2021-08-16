#include <gtest/gtest.h>

#include "umappp/spectral_init.hpp"

#include <random>
#include <vector>

class SpectralInitTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    template<class Param>
    void assemble(Param p) {
        // Creating a mock symmetric matrix.
        std::mt19937_64 rng(1234567890);
        std::uniform_real_distribution<> dist(0, 1);

        order = std::get<0>(p);
        ndim = std::get<1>(p);

        for (int r = 0; r < order; ++r) {
            for (int c = 0; c < r; ++c) {
                if (dist(rng) < 0.2) { // sparse lower-triangular matrix.
                    edges.emplace_back(r, c, dist(rng));
                }
            }
            edges.emplace_back(r, r, dist(rng)); // diagonal.
        }

        output.resize(ndim * order);
    }

    int order, ndim;
    umappp::Edges edges;
    std::vector<double> output;
};

TEST_P(SpectralInitTest, Basic) {
    assemble(GetParam());
    umappp::spectral_init(edges, order, ndim, output.data());
}

INSTANTIATE_TEST_SUITE_P(
    SpectralInit,
    SpectralInitTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(2, 3) // number of dimensions
    )
);
