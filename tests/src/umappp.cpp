#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
// Define before umappp includes.
#include "custom_parallel.h"
#endif

#include "umappp/initialize.hpp"
#include "knncolle/knncolle.hpp"

#include <map>
#include <random>
#include <cmath>

class UmapTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    void SetUp() {
        auto p = GetParam();
        nobs = std::get<0>(p);
        k = std::get<1>(p);

        std::mt19937_64 rng(nobs * k); // for some variety
        std::normal_distribution<> dist(0, 1);

        size_t total = nobs* ndim;
        data.resize(total);
        for (size_t r = 0; r < total; ++r) {
            data[r] = dist(rng);
        }

        builder.reset(new knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::EuclideanDistance<double, double> >()));
        auto index = builder->build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
        auto searcher = index->initialize();
        std::vector<int> indices;
        std::vector<double> distances;

        neighbors.resize(nobs);
        for (int i = 0; i < nobs; ++i) {
            searcher->search(i, k, &indices, &distances);
            size_t actual_k = indices.size();
            for (size_t x = 0; x < actual_k; ++x) {
                neighbors[i].emplace_back(indices[x], distances[x]);
            }
        }
        return;
    }

    int nobs, k;
    int ndim = 5;
    std::vector<double> data;
    std::shared_ptr<knncolle::Builder<int, double, double> > builder;
    umappp::NeighborList<int, double> neighbors;
};

TEST_P(UmapTest, Basic) {
    std::vector<double> output(nobs * ndim);
    auto status = umappp::initialize(neighbors, ndim, output.data(), umappp::Options());

    EXPECT_EQ(status.epoch(), 0);
    EXPECT_EQ(status.num_epochs(), 500);
    EXPECT_EQ(status.num_observations(), nobs);

    status.run();
    EXPECT_EQ(status.epoch(), 500);
    for (auto o : output){ 
        // Check that we don't get any weirdness.
        EXPECT_FALSE(std::isnan(o));
    }

    // Same results if we ran it from the top.
    {
        std::vector<double> copy(nobs * ndim);
        auto status2 = umappp::initialize(ndim, nobs, data.data(), *builder, ndim, copy.data(), [&]{
            umappp::Options opt;
            opt.num_neighbors = k;
            return opt;
        }());
        status2.run();
        EXPECT_EQ(copy, output);
    }

    // Same results if we started a little, and then ran the rest.
    {
        std::vector<double> copy(nobs * ndim);
        auto status_partial = umappp::initialize(neighbors, ndim, copy.data(), umappp::Options());
        status_partial.run(200);
        EXPECT_EQ(status_partial.epoch(), 200);
        EXPECT_NE(copy, output);

        std::vector<double> replacement(copy.size());
        status_partial.set_embedding(replacement.data());
        status_partial.run();
        EXPECT_EQ(status_partial.epoch(), 500);
        EXPECT_EQ(replacement, output);
    }

    // Same results with multiple threads.
    {
        umappp::Options opt;
        opt.num_neighbors = k;
        opt.num_threads = 3;

        {
            std::vector<double> copy(nobs * ndim);
            auto status = umappp::initialize(ndim, nobs, data.data(), *builder, ndim, copy.data(), opt);
            status.run();
            EXPECT_EQ(copy, output);
        }

        // Same results with multiple threads and parallel optimization enabled.
        opt.parallel_optimization = true;
        {
            std::vector<double> copy(nobs * ndim);
            auto status = umappp::initialize(neighbors, ndim, copy.data(), opt);
            status.run();
            EXPECT_EQ(copy, output);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Umap,
    UmapTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors
    )
);

TEST(UmapTest, SinglePrecision) {
    int nobs = 87;
    int k = 5;
    int ndim = 7;

    std::mt19937_64 rng(nobs * k + 1); 
    std::normal_distribution<float> dist(0, 1);
    size_t total = nobs * ndim;
    std::vector<float> data(total);
    for (size_t r = 0; r < total; ++r) {
        data[r] = dist(rng);
    }

    std::vector<float> output(nobs * 2);
    auto float_builder = knncolle::VptreeBuilder<int, float, float>(std::make_shared<knncolle::EuclideanDistance<float, float> >());
    auto status = umappp::initialize(ndim, nobs, data.data(), float_builder, 2, output.data(), umappp::Options());

    status.run();
    EXPECT_EQ(status.epoch(), 500);
    for (auto o : output){ 
        // Check that we don't get any weirdness.
        EXPECT_FALSE(std::isnan(o));
    }
}

TEST(UmapTest, EpochDecay) {
    EXPECT_EQ(umappp::internal::choose_num_epochs(-1, 1000), 500);
    EXPECT_LT(umappp::internal::choose_num_epochs(-1, 20000), 500);
    EXPECT_EQ(umappp::internal::choose_num_epochs(-1, 10000000), 201);
    EXPECT_EQ(umappp::internal::choose_num_epochs(1000, 1000), 1000);
    EXPECT_EQ(umappp::internal::choose_num_epochs(1000, 20000), 1000);
}
