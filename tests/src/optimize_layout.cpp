#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
// Define before umappp includes.
#include "custom_parallel.h"
#endif

#include "umappp/neighbor_similarities.hpp"
#include "umappp/combine_neighbor_sets.hpp"
#include "umappp/optimize_layout.hpp"
#include "knncolle/knncolle.hpp"

#include <vector>
#include <random>

class OptimizeTest : public ::testing::TestWithParam<std::tuple<int, int> > {
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

        auto index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
        auto searcher = index->initialize();
        std::vector<int> indices;
        std::vector<double> distances;

        stored.resize(nobs);
        for (int i = 0; i < nobs; ++i) {
            searcher->search(i, k, &indices, &distances);
            size_t actual_k = indices.size();
            for (size_t x = 0; x < actual_k; ++x) {
                stored[i].emplace_back(indices[x], distances[x]);
            }
        }

        umappp::internal::neighbor_similarities(stored);
        umappp::internal::combine_neighbor_sets(stored, 1.0);
        return;
    }

    int nobs, k;
    int ndim = 5;
    std::vector<double> data;
    std::vector<std::vector<std::pair<int, double> > > stored;
};

TEST_P(OptimizeTest, Epochs) {
    stored[0][0].second = 1e-8; // check for correct removal.

    auto epoch = umappp::internal::similarities_to_epochs(stored, 500, 5.0);
    EXPECT_EQ(epoch.head.size(), nobs);
    EXPECT_EQ(epoch.tail.size(), epoch.epochs_per_sample.size());
    EXPECT_EQ(epoch.tail.size(), epoch.head.back());

    // Make sure that we lost something.
    size_t total_n = 0;
    for (auto x : stored) {
        total_n += x.size();        
    }
    EXPECT_TRUE(total_n > epoch.epochs_per_sample.size());

    // All survivors should be no less than 1.
    for (auto x : epoch.epochs_per_sample) {
        EXPECT_TRUE(x >= 1);
    }
}

TEST_P(OptimizeTest, BasicRun) {
    auto epoch = umappp::internal::similarities_to_epochs(stored, 500, 5.0);

    std::vector<double> embedding(data);
    std::mt19937_64 rng(10);
    umappp::internal::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 0);

    EXPECT_NE(embedding, data); // some kind of change happened!
}

TEST_P(OptimizeTest, RestartedRun) {
    auto epoch = umappp::internal::similarities_to_epochs(stored, 500, 5.0);

    std::vector<double> embedding(data);
    std::mt19937_64 rng(10);
    umappp::internal::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 100);
    umappp::internal::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 500);

    // Same results from a full single run.
    std::vector<double> embedding2(data);
    rng.seed(10);
    epoch = umappp::internal::similarities_to_epochs(stored, 500, 5.0);
    umappp::internal::optimize_layout<>(5, embedding2.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 0);

    EXPECT_EQ(embedding, embedding2);
}

TEST_P(OptimizeTest, ParallelRun) {
    auto epoch = umappp::internal::similarities_to_epochs(stored, 500, 5.0);
    auto epoch2 = epoch;

    std::vector<double> embedding(data);
    {
        std::mt19937_64 rng(100);
        umappp::internal::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 0);
    }

    // Trying with more threads.
    std::vector<double> embedding2(data);
    {
        std::mt19937_64 rng(100);
        umappp::internal::optimize_layout_parallel<>(5, embedding2.data(), epoch2, 2.0, 1.0, 1.0, 1.0, rng, 0, 3);
    }

    EXPECT_NE(data, embedding); // some kind of change happened!
    EXPECT_EQ(embedding, embedding2); 
}

INSTANTIATE_TEST_SUITE_P(
    Optimize,
    OptimizeTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors
    )
);
