#include <gtest/gtest.h>

#include "umappp/neighbor_similarities.hpp"
#include "knncolle/knncolle.hpp"

#include <map>

class SimilarityTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    template<class Param>
    void assemble(Param p) {
        nobs = std::get<0>(p);
        k = std::get<1>(p);

        std::mt19937_64 rng(nobs * k); // for some variety
        std::normal_distribution<> dist(0, 1);

        data.resize(nobs * ndim);
        for (int r = 0; r < data.size(); ++r) {
            data[r] = dist(rng);
        }
    }

    auto generate_neighbors () { 
        knncolle::VpTreeEuclidean<> searcher(ndim, nobs, data.data());
        std::vector<std::vector<std::pair<int, double> > > stored;
        stored.reserve(nobs);
        for (size_t i = 0; i < searcher.nobs(); ++i) {
            stored.push_back(searcher.find_nearest_neighbors(i, k));
        }
        return stored;
    }

    int nobs, k;
    int ndim = 5;
    std::vector<double> data;
};

TEST_P(SimilarityTest, Basic) {
    assemble(GetParam());
    auto stored = generate_neighbors();
    
    umappp::neighbor_similarities(stored);

    for (const auto& s : stored) {
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
    assemble(GetParam());

    // Clone first vector into the second and third.
    std::copy(data.begin(), data.begin() + ndim, data.begin() + ndim);
    std::copy(data.begin(), data.begin() + ndim, data.begin() + ndim * 2);

    auto stored = generate_neighbors();
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(stored[i][0].second, 0);
        EXPECT_EQ(stored[i][1].second, 0);
    }

    // Distances of 0 map to weights of 1.
    umappp::neighbor_similarities(stored);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(stored[i][0].second, 1);
        EXPECT_EQ(stored[i][1].second, 1);
    }
}

TEST_P(SimilarityTest, EdgeCases) {
    assemble(GetParam());
    auto stored = generate_neighbors();

    // Forcing the fallback when local_connectivity is too high.
    umappp::neighbor_similarities(stored, 64, 100); 

    for (const auto& s : stored) {
        for (const auto& v : s){ 
            EXPECT_EQ(v.second, 1);
        }
    }

    // Clone first vector into the next 'k' neighbors; this results in all-zero distances.
    for (int i = 1; i < k; ++i) {
        std::copy(data.begin(), data.begin() + ndim, data.begin() + ndim * i);
    }
    stored = generate_neighbors();

    umappp::neighbor_similarities(stored);
    auto ref = stored[0];
    for (int i = 0; i < k; ++i) {
        for (auto j : stored[i]) {
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
