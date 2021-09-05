#include <gtest/gtest.h>

#include "umappp/spectral_init.hpp"

#include <random>
#include <vector>

class SpectralInitTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    static umappp::NeighborList mock(int n) {
        // Creating a mock symmetric matrix.
        std::mt19937_64 rng(1234567890);
        std::uniform_real_distribution<> dist(0, 1);

        umappp::NeighborList edges(n);
        edges.resize(n);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < r; ++c) {
                if (dist(rng) < 0.2) { // sparse symmetric matrix.
                    double val = dist(rng);
                    edges[r].emplace_back(c, val);
                    edges[c].emplace_back(r, val);
                }
            }
            edges[r].emplace_back(r, dist(rng)); // diagonal.
        }
        return edges;
    }
};

TEST_P(SpectralInitTest, Basic) {
    auto p = GetParam();
    int order = std::get<0>(p);
    int ndim = std::get<1>(p);
    std::vector<double> output(ndim * order);

    auto edges = mock(order);
    umappp::spectral_init(edges, ndim, output.data());

    for (auto o : output) { // filled with _something_.
        EXPECT_TRUE(o != 0);
    }
}

TEST_P(SpectralInitTest, MultiComponents) {
    auto p = GetParam();
    int order = std::get<0>(p);
    int ndim = std::get<1>(p);

    auto edges1 = mock(order);
    std::vector<double> out1(edges1.size() * ndim);
    umappp::spectral_init(edges1, ndim, out1.data());

    auto edges2 = mock(order * 2);
    std::vector<double> out2(edges2.size() * ndim);
    umappp::spectral_init(edges2, ndim, out2.data());

    // Combining the components.
    auto edges = edges1;
    for (const auto& e : edges2) {
        edges.push_back(e);
        for (auto& x : edges.back()) {
            x.first += order; // adjusting the neighbor index.
        }
    }

    std::vector<double> output(out1.size() + out2.size());
    umappp::spectral_init(edges, ndim, output.data());

    EXPECT_EQ(out1, std::vector<double>(output.begin(), output.begin() + out1.size()));
    EXPECT_EQ(out2, std::vector<double>(output.begin() + out1.size(), output.end()));
}

TEST_P(SpectralInitTest, MultiComponentsInterspersed) {
    auto p = GetParam();
    int order = std::get<0>(p);
    int ndim = std::get<1>(p);

    auto edges1 = mock(order);
    std::vector<double> out1(edges1.size() * ndim);
    umappp::spectral_init(edges1, ndim, out1.data());

    auto edges2 = mock(order);
    std::vector<double> out2(edges2.size() * ndim);
    umappp::spectral_init(edges2, ndim, out2.data());

    auto edges3 = mock(order);
    std::vector<double> out3(edges3.size() * ndim);
    umappp::spectral_init(edges3, ndim, out3.data());

    // Interspersing the components.
    umappp::NeighborList edges;
    for (int o = 0; o < order; ++o) {
        edges.push_back(edges1[o]);
        for (auto& e : edges.back()) {
            e.first *= 3;
        }

        edges.push_back(edges2[o]);
        for (auto& e : edges.back()) {
            e.first *= 3;
            e.first += 1;
        }

        edges.push_back(edges3[o]);
        for (auto& e : edges.back()) {
            e.first *= 3;
            e.first += 2;
        }
    }

    std::vector<double> output(out1.size() + out2.size() + out3.size());
    umappp::spectral_init(edges, ndim, output.data());

    for (int o = 0; o < order; ++o) {
        EXPECT_EQ(
            std::vector<double>(out1.begin() + o * ndim, out1.begin() + (o + 1) * ndim), 
            std::vector<double>(output.begin() + o * 3 * ndim, output.begin() + (o * 3 + 1) * ndim)
        );
        EXPECT_EQ(
            std::vector<double>(out2.begin() + o * ndim, out2.begin() + (o + 1) * ndim), 
            std::vector<double>(output.begin() + (o * 3 + 1) * ndim, output.begin() + (o * 3 + 2) * ndim)
        );
        EXPECT_EQ(
            std::vector<double>(out3.begin() + o * ndim, out3.begin() + (o + 1) * ndim), 
            std::vector<double>(output.begin() + (o * 3 + 2) * ndim, output.begin() + (o * 3 + 3) * ndim)
        );
    }
}

INSTANTIATE_TEST_SUITE_P(
    SpectralInit,
    SpectralInitTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(2, 5) // number of dimensions
    )
);
