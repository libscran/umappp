#include <gtest/gtest.h>

#include "umappp/neighbor_similarities.hpp"
#include "knncolle/knncolle.hpp"

#include <map>

class SimilarityTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    void SetUp() {
        auto p = GetParam();
        nobs = std::get<0>(p);
        k = std::get<1>(p);

        std::mt19937_64 rng(nobs * k); // for some variety
        std::normal_distribution<> dist(0, 1);

        data.resize(nobs * ndim);
        for (size_t r = 0; r < data.size(); ++r) {
            data[r] = dist(rng);
        }
    }

    int nobs, k;
    int ndim = 5;
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

TEST_P(SimilarityTest, Basic) {
    auto neighbors = generate_neighbors(ndim, nobs, data, k);
    umappp::internal::neighbor_similarities(neighbors);

    for (const auto& s : neighbors) {
        double prev = 1;
        bool first = true;
        for (const auto& v : s){ 
            if (first) {
                EXPECT_EQ(v.second, 1);
                first = false;
            } else {
                EXPECT_TRUE(v.second < 1);
            }
            EXPECT_TRUE(v.second <= prev); // should be decreasing
            prev = v.second;
        }
    }
}

TEST_P(SimilarityTest, ZeroDistance) {
    // Clone first vector into the second and third.
    std::copy(data.begin(), data.begin() + ndim, data.begin() + ndim);
    std::copy(data.begin(), data.begin() + ndim, data.begin() + ndim * 2);

    auto neighbors = generate_neighbors(ndim, nobs, data, k);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(neighbors[i][0].second, 0);
        EXPECT_EQ(neighbors[i][1].second, 0);
    }

    // Distances of 0 map to weights of 1.
    umappp::internal::neighbor_similarities(neighbors);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(neighbors[i][0].second, 1);
        EXPECT_EQ(neighbors[i][1].second, 1);
    }
}

TEST_P(SimilarityTest, EdgeCases) {
    auto neighbors = generate_neighbors(ndim, nobs, data, k);

    // Forcing the fallback when local_connectivity is too high.
    umappp::internal::neighbor_similarities(neighbors, 100.0); 

    for (const auto& s : neighbors) {
        for (const auto& v : s){ 
            EXPECT_EQ(v.second, 1);
        }
    }

    // Clone first vector into the next 'k' neighbors; this results in all-zero distances.
    for (int i = 1; i < k; ++i) {
        std::copy(data.begin(), data.begin() + ndim, data.begin() + ndim * i);
    }
    neighbors = generate_neighbors(ndim, nobs, data, k);

    umappp::internal::neighbor_similarities(neighbors);
    auto ref = neighbors[0];
    for (int i = 0; i < k; ++i) {
        for (auto j : neighbors[i]) {
            EXPECT_EQ(j.second, 1);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Similarity,
    SimilarityTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors 
    )
);
