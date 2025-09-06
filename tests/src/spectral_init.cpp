#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
// Define before umappp includes.
#include "custom_parallel.h"
#endif

#include "umappp/spectral_init.hpp"

#include <random>
#include <vector>

static umappp::NeighborList<int, double> mock(int n) {
    // Creating a mock symmetric matrix.
    std::mt19937_64 rng(1234567890);
    std::uniform_real_distribution<> dist(0, 1);

    umappp::NeighborList<int, double> edges(n);
    edges.resize(n);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < r; ++c) {
            if (dist(rng) < 0.2) { // sparse symmetric matrix.
                double val = dist(rng);
                edges[r].emplace_back(c, val);
                edges[c].emplace_back(r, val);
            }
        }
    }
    return edges;
}

class SpectralInitTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    static constexpr double max_scale = 10;
    static constexpr int seed = 12345;
    static constexpr double jitter_sd = 0.0001;
};

TEST_P(SpectralInitTest, Basic) {
    auto p = GetParam();
    int order = std::get<0>(p);
    int ndim = std::get<1>(p);
    std::vector<double> output(ndim * order);

    auto edges = mock(order);
    EXPECT_TRUE(umappp::internal::spectral_init(edges, ndim, output.data(), 1, max_scale, false, jitter_sd, seed));

    for (auto o : output) { // filled with _something_.
        EXPECT_TRUE(o != 0);
    }
    const double max_val = std::max(*std::max_element(output.begin(), output.end()), -*std::min_element(output.begin(), output.end()));
    EXPECT_FLOAT_EQ(max_val, max_scale);

    // Same result with multiple threads.
    std::vector<double> copy(ndim * order);
    umappp::internal::spectral_init(edges, ndim, copy.data(), 3, max_scale, false, jitter_sd, seed);
    EXPECT_EQ(output, copy);

    // Throwing in some jitter.
    std::fill(copy.begin(), copy.end(), 0);
    EXPECT_TRUE(umappp::internal::spectral_init(edges, ndim, copy.data(), 1, max_scale, true, jitter_sd, seed));
    EXPECT_NE(output, copy);
}

TEST_P(SpectralInitTest, MultiComponents) {
    auto p = GetParam();
    int order = std::get<0>(p);
    int ndim = std::get<1>(p);

    auto edges1 = mock(order);
    auto edges2 = mock(order * 2);

    // Combining the components.
    auto edges = edges1;
    for (const auto& e : edges2) {
        edges.push_back(e);
        for (auto& x : edges.back()) {
            x.first += order; // adjusting the neighbor index.
        }
    }

    std::vector<double> output(edges1.size() + edges2.size());
    EXPECT_FALSE(umappp::internal::spectral_init(edges, ndim, output.data(), 1, max_scale, false, jitter_sd, seed));
}

INSTANTIATE_TEST_SUITE_P(
    SpectralInit,
    SpectralInitTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(2, 5) // number of dimensions
    )
);

static void symmetrize(umappp::NeighborList<int, double>& x) {
    std::vector<size_t> available;
    available.reserve(x.size());
    for (const auto& y : x) {
        available.push_back(y.size());
    }

    for (size_t i = 0; i < x.size(); ++i) {
        auto& current = x[i];
        for (size_t j = 0; j < available[i]; ++j) {
            x[current[j].first].emplace_back(i, current[j].second);
        }
    }
    return;
}

TEST(ComponentTest, Simple) {
    int order = 5;

    umappp::NeighborList<int, double> edges(order);
    edges[4].emplace_back(0, 0.5);
    edges[4].emplace_back(1, 0.5);
    edges[3].emplace_back(2, 0.5);

    auto copy = edges;
    symmetrize(copy);
    EXPECT_TRUE(umappp::internal::has_multiple_components(copy));

    // Merging into one component.
    edges[3].emplace_back(1, 0.5);

    copy = edges;
    symmetrize(copy);
    EXPECT_FALSE(umappp::internal::has_multiple_components(copy));

    {
        int order = 5;
        umappp::NeighborList<int, double> edges(order);
        EXPECT_TRUE(umappp::internal::has_multiple_components(edges));

        // Sticking in an edge to merge nodes.
        edges[3].emplace_back(1, 0.5);

        auto copy = edges;
        symmetrize(copy);
        EXPECT_TRUE(umappp::internal::has_multiple_components(copy));
    }
        
    {
        int order = 6;

        // Deliberately checking the case where one node splits into two,
        // or two nodes merge into one, depending on whether we are 
        // traversing in order of increasing or decreasing index.
        umappp::NeighborList<int, double> edges(order);
        edges[4].emplace_back(2, 0.5);
        edges[4].emplace_back(3, 0.5);

        edges[5].emplace_back(1, 0.5);
        edges[5].emplace_back(0, 0.5);

        symmetrize(edges);
        EXPECT_TRUE(umappp::internal::has_multiple_components(edges));
    }
}

TEST(SpectralInit, OddJitter) { // test coverage when the number of coordinates is odd.
    auto edges = mock(51);
    int ndim = 3;
    std::vector<double> output(edges.size() * ndim);
    EXPECT_TRUE(output.size() % 2 == 1);

    EXPECT_TRUE(umappp::internal::spectral_init(edges, ndim, output.data(), 1, 10, true, 0.001, 69));
    for (auto o : output) {
        EXPECT_NE(o, 0);
        EXPECT_GE(o, -10);
        EXPECT_LE(o, 10);
    }
}

TEST(RandomInit, Basic) {
    std::vector<double> output(15);
    umappp::internal::random_init(5, 3, output.data(), 69, 10);
    for (auto o : output) {
        EXPECT_NE(o, 0); // filled with _something_.
        EXPECT_GE(o, -10);
        EXPECT_LT(o, 10);
    }
}
