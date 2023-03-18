#include <gtest/gtest.h>

#ifndef TEST_NUM_THREADS
#define TEST_NUM_THREADS 3
#endif

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

        knncolle::VpTreeEuclidean<> searcher(ndim, nobs, data.data());
        stored.reserve(nobs);
        for (size_t i = 0; i < searcher.nobs(); ++i) {
            stored.push_back(searcher.find_nearest_neighbors(i, k));
        }

        umappp::neighbor_similarities(stored);
        umappp::combine_neighbor_sets(stored, 1.0);
        return;
    }

    int nobs, k;
    int ndim = 5;
    std::vector<double> data;
    std::vector<std::vector<std::pair<int, double> > > stored;
};

TEST_P(OptimizeTest, Epochs) {
    assemble(GetParam());
    stored[0][0].second = 1e-8; // check for correct removal.

    auto epoch = umappp::similarities_to_epochs(stored, 500, 5.0);
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
    assemble(GetParam());
    auto epoch = umappp::similarities_to_epochs(stored, 500, 5.0);

    std::vector<double> embedding(data);
    std::mt19937_64 rng(10);
    umappp::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 0);

    EXPECT_NE(embedding, data); // some kind of change happened!
}

TEST_P(OptimizeTest, RestartedRun) {
    assemble(GetParam());
    auto epoch = umappp::similarities_to_epochs(stored, 500, 5.0);

    std::vector<double> embedding(data);
    std::mt19937_64 rng(10);
    umappp::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 100);
    umappp::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 500);

    // Same results from a full single run.
    std::vector<double> embedding2(data);
    rng.seed(10);
    epoch = umappp::similarities_to_epochs(stored, 500, 5.0);
    umappp::optimize_layout<>(5, embedding2.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 0);

    EXPECT_EQ(embedding, embedding2);
}

TEST_P(OptimizeTest, ParallelRun) {
    assemble(GetParam());
    auto epoch = umappp::similarities_to_epochs(stored, 500, 5.0);
    auto epoch2 = epoch;

    std::vector<double> embedding(data);
    {
        std::mt19937_64 rng(100);
        umappp::optimize_layout<>(5, embedding.data(), epoch, 2.0, 1.0, 1.0, 1.0, rng, 0);
    }

    // Trying with more threads.
    std::vector<double> embedding2(data);
    {
        std::mt19937_64 rng(100);
        umappp::optimize_layout_parallel<>(5, embedding2.data(), epoch2, 2.0, 1.0, 1.0, 1.0, rng, 0, TEST_NUM_THREADS);
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
