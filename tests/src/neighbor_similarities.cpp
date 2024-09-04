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

        std::mt19937_64 rng(nobs * k); // for some variety
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
        std::vector<std::vector<std::pair<int, double> > > neighbors(nobs);
        auto index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
        auto searcher = index->initialize();
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
    umappp::internal::neighbor_similarities(neighbors, connectivity);

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

TEST_P(SimilarityTest, BinarySearch) {
    auto neighbors = generate_neighbors(ndim, nobs, data, k);
    umappp::internal::neighbor_similarities<false>(neighbors, connectivity); 

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
        ::testing::Values(5, 10, 15), // number of neighbors 
        ::testing::Values(0.4, 1, 1.3, 2, 2.5) // local connectivity.
    )
);

TEST(NeighborSimilarities, Empty) {
    umappp::NeighborList<int, double> neighbors(1);
    umappp::internal::neighbor_similarities(neighbors);
    EXPECT_TRUE(neighbors.front().empty());
}

TEST(NeighborSimilarities, AllZeroDistance) {
    umappp::NeighborList<int, double> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
    }

    umappp::internal::neighbor_similarities(neighbors);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_EQ(s.second, 1);
        }
    }
}

TEST(NeighborSimilarities, NoAboveRho) {
    // We add tied distances to force everything to be <= rho.
    umappp::NeighborList<int, double> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
        for (auto& x : neighbors[i]) {
            x.second = 10.0;
        }
    }

    umappp::internal::neighbor_similarities(neighbors);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_EQ(s.second, 1);
        }
    }
}

TEST(NeighborSimilarities, TooHighConnectivity) {
    // Forcing the fallback when local_connectivity is too high.
    umappp::NeighborList<int, double> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
        for (size_t j = 0; j < neighbors[i].size(); ++j) {
            neighbors[i][j].second = j * 0.1;
        }
    }

    umappp::internal::neighbor_similarities(neighbors, 100.0);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_EQ(s.second, 1);
        }
    }
}

TEST(NeighborSimilarities, ConvergenceFailure) {
    // Setting the bandwidth to be zero so that it's impossible to get there.
    // The aim is then to perform enough iterations so we end up with 'sigma'
    // very close to zero, which causes the protection to kick in.
    umappp::NeighborList<int, float> neighbors(3);
    for (int i = 0; i < 3; ++i) {
        neighbors[i].resize(20);
        for (size_t j = 0; j < neighbors[i].size(); ++j) {
            neighbors[i][j].second = j * 0.1;
        }
    }

    umappp::internal::neighbor_similarities<false, int, float>(neighbors, 1.0, 0.0, /* max_iter = */ 200);
    for (int i = 0; i < 3; ++i) {
        for (auto s : neighbors[i]) {
            EXPECT_LE(s.second, 1);
        }
        EXPECT_GT(neighbors[i].front().second, neighbors[i].back().second); 
    }
}

