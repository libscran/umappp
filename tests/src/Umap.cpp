#include <gtest/gtest.h>

#ifndef TEST_NUM_THREADS
#define TEST_NUM_THREADS 3
#endif

#ifdef TEST_CUSTOM_PARALLEL
// Define before umappp includes.
#include "custom_parallel.h"
#endif

#include "umappp/Umap.hpp"
#include "knncolle/knncolle.hpp"

#include <map>
#include <random>
#include <cmath>

class UmapTest : public ::testing::TestWithParam<std::tuple<int, int> > {
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
        return;
    }

    int nobs, k;
    int ndim = 5;
    std::vector<double> data;
    umappp::NeighborList<double> stored;
};

TEST_P(UmapTest, Basic) {
    auto param = GetParam();
    assemble(param);

    umappp::Umap<> runner;

    std::vector<double> output(nobs * ndim);
    auto status = runner.initialize(std::move(stored), ndim, output.data());
    EXPECT_EQ(status.epoch(), 0);
    EXPECT_EQ(status.num_epochs(), 500);
    EXPECT_EQ(status.nobs(), nobs);

    status.run();
    EXPECT_EQ(status.epoch(), 500);

    // Same results if we ran it from the top.
    std::vector<double> copy(nobs * ndim);
    runner.set_num_neighbors(k);
    runner.run(ndim, nobs, data.data(), ndim, copy.data());
    EXPECT_EQ(copy, output);

    // Same results with multiple threads.
    runner.set_num_threads(TEST_NUM_THREADS);
    std::fill(copy.begin(), copy.end(), 0);
    runner.run(ndim, nobs, data.data(), ndim, copy.data());
    EXPECT_EQ(copy, output);

    // Same results with multiple threads and parallel optimization enabled.
    runner.set_parallel_optimization(true);
    std::fill(copy.begin(), copy.end(), 0);
    runner.run(ndim, nobs, data.data(), ndim, copy.data());
    EXPECT_EQ(copy, output);
}

INSTANTIATE_TEST_SUITE_P(
    Umap,
    UmapTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors
    )
);

TEST(UmapTest, EpochDecay) {
    EXPECT_EQ(umappp::choose_num_epochs(-1, 1000), 500);
    EXPECT_TRUE(umappp::choose_num_epochs(-1, 20000) < 500);
    EXPECT_EQ(umappp::choose_num_epochs(-1, 10000000), 201);
    EXPECT_EQ(umappp::choose_num_epochs(1000, 1000), 1000);
    EXPECT_EQ(umappp::choose_num_epochs(1000, 20000), 1000);
}
