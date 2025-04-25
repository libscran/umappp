#include <gtest/gtest.h>

#include "umappp/neighbor_similarities.hpp"
#include "knncolle/knncolle.hpp"

#include <map>

class SimilarityTest : public ::testing::TestWithParam<std::tuple<int, int, double> > {
protected:
    void SetUp() {
        auto p = GetParam();
        nobs = std::get<0>(p);
        k = std::get<1>(p);
        connectivity = std::get<2>(p);

        std::mt19937_64 rng(nobs * k + connectivity * 10); // for some variety
        std::normal_distribution<> dist(0, 1);

        data.resize(nobs * ndim);
        for (size_t r = 0; r < data.size(); ++r) {
            data[r] = dist(rng);
        }
    }

    int nobs, k;
    int ndim = 5;
    double connectivity;
    std::vector<double> data;

protected:
    static auto generate_neighbors(int ndim, int nobs, const std::vector<double>& data, int k) {
        auto builder = knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::EuclideanDistance<double, double> >());
        auto index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
        auto searcher = index->initialize();

        std::vector<std::vector<std::pair<int, double> > > neighbors(nobs);
        std::vector<int> indices;
        std::vector<double> distances;

        for (int i = 0; i < nobs; ++i) {
            searcher->search(i, k, &indices, &distances);
            size_t actual_k = indices.size();
            for (size_t x = 0; x < actual_k; ++x) {
                neighbors[i].emplace_back(indices[x], distances[x]);
            }
        }

        return neighbors;
    }
};

TEST_P(SimilarityTest, Newton) {
    auto neighbors = generate_neighbors(ndim, nobs, data, k);
    auto copy = neighbors;

    umappp::internal::NeighborSimilaritiesOptions<double> opts;
    opts.local_connectivity = connectivity;
    opts.min_k_dist_scale = 1e-8; // turn off protection for the time being.
    umappp::internal::neighbor_similarities(neighbors, opts);

    for (const auto& s : neighbors) {
        double prev = 1;
        double observed = 0;

        for (size_t i = 0; i < s.size(); ++i) {
            const auto& v = s[i];
            if (i < static_cast<size_t>(connectivity)) {
                EXPECT_EQ(v.second, 1);
            } else {
                EXPECT_LT(v.second, 1);
            }
            EXPECT_LE(v.second, prev); // should be decreasing
            prev = v.second;
            observed += v.second;
        }

        // Checking for proper convergence.
        double expected = std::log2(s.size() + 1);
        EXPECT_LT(std::abs(observed - expected), 1e-5);
    }

    // Same results in parallel.
    opts.num_threads = 3;
    umappp::internal::neighbor_similarities(copy, opts);
    for (int i = 0; i < nobs; ++i) {
        EXPECT_EQ(copy[i], neighbors[i]);
    }
}

TEST_P(SimilarityTest, BinarySearch) {
    auto neighbors = generate_neighbors(ndim, nobs, data, k);

    umappp::internal::NeighborSimilaritiesOptions<double> opts;
    opts.local_connectivity = connectivity;
    opts.min_k_dist_scale = 1e-8; // turn off protection for the time being.
    umappp::internal::neighbor_similarities<false>(neighbors, opts); 

    for (const auto& s : neighbors) {
        double prev = 1;
        double observed = 0;

        for (size_t i = 0; i < s.size(); ++i) {
            const auto& v = s[i];
            if (i < static_cast<size_t>(connectivity)) {
                EXPECT_EQ(v.second, 1);
            } else {
                EXPECT_LT(v.second, 1);
            }
            EXPECT_LE(v.second, prev); // should be decreasing
            prev = v.second;
            observed += v.second;
        }

        // Checking for proper convergence.
        double expected = std::log2(s.size() + 1);
        EXPECT_LT(std::abs(observed - expected), 1e-5);
    }
}

INSTANTIATE_TEST_SUITE_P(
    NeighborSimilarities,
    SimilarityTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 20), // number of neighbors 
        ::testing::Values(0.4, 1, 1.3, 2, 2.5) // local connectivity.
    )
);

TEST(NeighborSimilarities, Empty) {
    umappp::NeighborList<int, double> neighbors(1);
    umappp::internal::NeighborSimilaritiesOptions<double> opts;
    umappp::internal::neighbor_similarities(neighbors, opts);
    EXPECT_TRUE(neighbors.front().empty());
}

TEST(NeighborSimilarities, AllZeroDistance) {
    // Forcing an early quit via the all-zero condition.
    umappp::NeighborList<int, double> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
    }

    umappp::internal::NeighborSimilaritiesOptions<double> opts;
    umappp::internal::neighbor_similarities(neighbors, opts);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_EQ(s.second, 1);
        }
    }
}

TEST(NeighborSimilarities, NoAboveRho) {
    // Forcing an early quit by adding ties so that all distances <= rho.
    umappp::NeighborList<int, double> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
        for (auto& x : neighbors[i]) {
            x.second = 10.0;
        }
    }

    umappp::internal::NeighborSimilaritiesOptions<double> opts;
    umappp::internal::neighbor_similarities(neighbors, opts);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_EQ(s.second, 1);
        }
    }
}

TEST(NeighborSimilarities, TooHighConnectivity) {
    // Forcing an early quit when local_connectivity is too high.
    umappp::NeighborList<int, double> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
        for (size_t j = 0; j < neighbors[i].size(); ++j) {
            neighbors[i][j].second = j * 0.1;
        }
    }

    umappp::internal::NeighborSimilaritiesOptions<double> opts;
    opts.local_connectivity = 100.0;
    umappp::internal::neighbor_similarities(neighbors, opts);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_EQ(s.second, 1);
        }
    }
}

TEST(NeighborSimilarities, BoundedSigma) {
    // Setting the bandwidth to be zero so that it's impossible to get there.
    // The aim is then to perform enough iterations so we end up with 'sigma'
    // very close to zero, which causes the protection to kick in.
    umappp::NeighborList<int, float> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
        for (size_t j = 0; j < neighbors[i].size(); ++j) {
            neighbors[i][j].second = j * 0.1 + 0.01;
        }
    }
    auto copy = neighbors;

    umappp::internal::NeighborSimilaritiesOptions<float> opts;
    opts.bandwidth = 0;
    opts.min_k_dist_scale = 0.1;
    umappp::internal::neighbor_similarities<false, int, float>(neighbors, opts);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_LE(s.second, 1);
            EXPECT_GT(s.second, 0); // protection ensures that we don't end up with exp(-BIG_NUMBER).
        }
    }

    // If the protection is disabled, all values but the first are equal to 0 as sigma is too small.
    opts.min_k_dist_scale = 0;
    umappp::internal::neighbor_similarities<false, int, float>(copy, opts);
    for (int i = 0; i < 3; ++i) {
        for (size_t j = 0; j < copy[i].size(); ++j) {
            if (j == 0) {
                EXPECT_EQ(copy[i][j].second, 1);
            } else {
                EXPECT_EQ(copy[i][j].second, 0);
            }
        }
    }
}
