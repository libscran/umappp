#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
// Define before umappp includes.
#include "custom_parallel.h"
#endif

#include "umappp/spectral_init.hpp"

#include <random>
#include <vector>
#include <algorithm>

static umappp::NeighborList<int, double> mock_probabilities(int n) {
    // Mocking a sparse symmetric matrix of probabilities,
    // akin to that produced by umappp::internal::combine_neighbor_sets().
    std::mt19937_64 rng(n);

    umappp::NeighborList<int, double> edges(n);
    for (int r = 0; r < n; ++r) {
        // Forcibly connecting it to the previous observation, or wrapping around to the last observation.
        // This ensures that we only have 1 component in the graph.
        const int forced = (r == 0 ? n : r) - 1; 
        for (int c = 0; c < r; ++c) {
            if (c == forced || aarand::standard_uniform(rng) < 0.2) {
                double val = aarand::standard_uniform(rng);
                edges[r].emplace_back(c, val);
                edges[c].emplace_back(r, val);
            }
        }
    }

    return edges;
}

static void check_eigenvectors(const umappp::NeighborList<int, double>& probabilities, const std::vector<double>& results, int ndim) {
    const int order = probabilities.size();
    std::vector<double> sums(order);
    for (int o = 0; o < order; ++o) {
        const auto& pp = probabilities[o];
        for (const auto& x : pp) {
            sums[o] += x.second;
        }
    }
    for (auto& ss : sums) {
        ss = std::sqrt(ss);
    }

    // Compute product with all eigenvectors.
    std::vector<double> copy(results.size());
    for (int o = 0; o < order; ++o) {
        const auto& pp = probabilities[o];
        for (const auto& x : pp) {
            const auto norm_lap = (- x.second / sums[o] / sums[x.first]);
            for (int d = 0; d < ndim; ++d) {
                const auto in_offset = sanisizer::nd_offset<std::size_t>(d, ndim, x.first);
                const auto out_offset = sanisizer::nd_offset<std::size_t>(d, ndim, o);
                copy[out_offset] += results[in_offset] * norm_lap;
            }
        }
        for (int d = 0; d < ndim; ++d) {
            const auto offset = sanisizer::nd_offset<std::size_t>(d, ndim, o);
            copy[offset] += results[offset];
        }
    }

    // Check that it's a simple scaling difference.
    std::vector<double> res_scale(ndim), copy_scale(ndim);
    for (int o = 0; o < order; ++o) {
        for (int d = 0; d < ndim; ++d) {
            const auto offset = sanisizer::nd_offset<std::size_t>(d, ndim, o);
            res_scale[d] += results[offset];
            copy_scale[d] += copy[offset];
        }
    }

    for (int o = 0; o < order; ++o) {
        for (int d = 0; d < ndim; ++d) {
            const auto offset = sanisizer::nd_offset<std::size_t>(d, ndim, o);
            const auto refval = results[offset];
            const auto copyval = copy[offset] * res_scale[d] / copy_scale[d];
            EXPECT_LT(std::abs(refval - copyval), 0.0001 * (std::abs(refval) + std::abs(copyval)) / 2);
        }
    }
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

    irlba::Options iopt;
    iopt.convergence_tolerance = 1e-8; // improve accuracy for eigenvector check.

    auto edges = mock_probabilities(order);
    EXPECT_TRUE(umappp::internal::spectral_init(edges, ndim, output.data(), iopt, 1, max_scale, false, jitter_sd, seed));

    for (auto o : output) { // filled with _something_.
        EXPECT_TRUE(o != 0);
    }
    const double max_val = std::max(*std::max_element(output.begin(), output.end()), -*std::min_element(output.begin(), output.end()));
    EXPECT_FLOAT_EQ(max_val, max_scale);

    // Checking that the output actually contains eigenvectors for the normalized Laplacian.
    check_eigenvectors(edges, output, ndim);

    // Same result with multiple threads.
    std::vector<double> copy(ndim * order);
    umappp::internal::spectral_init(edges, ndim, copy.data(), iopt, 3, max_scale, false, jitter_sd, seed);
    EXPECT_EQ(output, copy);

    // Throwing in some jitter.
    std::fill(copy.begin(), copy.end(), 0);
    EXPECT_TRUE(umappp::internal::spectral_init(edges, ndim, copy.data(), iopt, 1, max_scale, true, jitter_sd, seed));
    EXPECT_NE(output, copy);
}

TEST_P(SpectralInitTest, MultiComponents) {
    auto p = GetParam();
    int order = std::get<0>(p);
    int ndim = std::get<1>(p);

    auto edges1 = mock_probabilities(order);
    auto edges2 = mock_probabilities(order * 2);

    // Combining the components.
    auto edges = edges1;
    for (const auto& e : edges2) {
        edges.push_back(e);
        for (auto& x : edges.back()) {
            x.first += order; // adjusting the neighbor index.
        }
    }

    std::vector<double> output(edges1.size() + edges2.size());
    EXPECT_FALSE(umappp::internal::spectral_init(edges, ndim, output.data(), irlba::Options{}, 1, max_scale, false, jitter_sd, seed));
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
    auto edges = mock_probabilities(51);
    int ndim = 3;
    std::vector<double> output(edges.size() * ndim);
    EXPECT_TRUE(output.size() % 2 == 1);

    const irlba::Options iopt;
    const int nthreads = 1;
    const double scale = 10;
    const double jitter_sd = 0.001;
    const int jitter_seed = 69;

    EXPECT_TRUE(umappp::internal::spectral_init(edges, ndim, output.data(), iopt, nthreads, scale, true, jitter_sd, jitter_seed));
    for (auto o : output) {
        EXPECT_NE(o, 0);
    }

    // Comparing it against the no-jitter reference, especially for the last entry.
    std::vector<double> ref(edges.size() * ndim);
    EXPECT_TRUE(umappp::internal::spectral_init(edges, ndim, ref.data(), iopt, nthreads, scale, false, jitter_sd, jitter_seed));
    EXPECT_NE(ref, output);
    EXPECT_NE(ref.back(), output.back());
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
