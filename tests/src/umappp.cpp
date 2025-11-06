#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
// Define before umappp includes.
#include "custom_parallel.h"
#endif

#include "umappp/initialize.hpp"
#include "knncolle/knncolle.hpp"
#include "aarand/aarand.hpp"

#include <random>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

class UmapTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    void SetUp() {
        auto p = GetParam();
        nobs = std::get<0>(p);
        k = std::get<1>(p);

        std::mt19937_64 rng(nobs * k); // for some variety
        std::uniform_real_distribution<> dist(0, 1);

        data.resize(nobs * ndim);
        for (int o = 0; o < nobs; ++o) {
            double offset = (o % 2 == 1 ? 10 : 0); // creating two populations that are shifted on the first dimension.
            for (int d = 0; d < ndim; ++d) {
                data[d + o * ndim] = dist(rng) + (d == 0 ? offset : 0);
            }
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
    int outdim = 2;
    std::vector<double> output(nobs * outdim);
    auto status = umappp::initialize(neighbors, outdim, output.data(), umappp::Options());

    EXPECT_EQ(status.epoch(), 0);
    EXPECT_EQ(status.num_epochs(), 500);
    EXPECT_EQ(status.num_observations(), nobs);

    status.run(output.data());
    EXPECT_EQ(status.epoch(), 500);
    for (auto o : output){ 
        // Check that we don't get any weirdness.
        EXPECT_FALSE(std::isnan(o));
    }

    // Sanity check for separation between the two groups of observations on at least one dimension.
    std::vector<double> min_odd(outdim, std::numeric_limits<double>::infinity());
    std::vector<double> max_odd(outdim, -std::numeric_limits<double>::infinity());
    auto min_even = min_odd;
    auto max_even = max_odd;
    for (int i = 0; i < nobs; ++i) {
        auto& min = (i % 2 == 0 ? min_even : min_odd);
        auto& max = (i % 2 == 0 ? max_even : max_odd);
        for (int d = 0; d < outdim; ++d) {
            auto val = output[outdim * i + d];
            min[d] = std::min(min[d], val);
            max[d] = std::max(max[d], val);
        }
    }
    EXPECT_TRUE(max_even[0] < min_odd[0] || max_odd[0] < min_even[0] || max_even[1] < min_odd[1] || max_odd[1] < min_even[1]);

    // Same results if we ran it from the top.
    {
        std::vector<double> copy(nobs * outdim);
        auto status2 = umappp::initialize(ndim, nobs, data.data(), *builder, outdim, copy.data(), [&]{
            umappp::Options opt;
            opt.num_neighbors = k;
            return opt;
        }());
        status2.run(copy.data());
        EXPECT_EQ(copy, output);
    }

    // Same results if we started a little, and then ran the rest.
    {
        std::vector<double> copy(nobs * outdim);
        auto status_partial = umappp::initialize(neighbors, outdim, copy.data(), umappp::Options());
        status_partial.run(copy.data(), 200);
        EXPECT_EQ(status_partial.epoch(), 200);
        EXPECT_NE(copy, output);

        status_partial.run(copy.data());
        EXPECT_EQ(status_partial.epoch(), 500);
        EXPECT_EQ(copy, output);
    }

    // Same results with multiple threads.
    {
        umappp::Options opt;
        opt.num_neighbors = k;
        opt.num_threads = 3;

        {
            std::vector<double> copy(nobs * outdim);
            auto status = umappp::initialize(ndim, nobs, data.data(), *builder, outdim, copy.data(), opt);
            status.run(copy.data());
            EXPECT_EQ(copy, output);
        }

        // Same results with multiple threads and parallel optimization enabled.
        opt.parallel_optimization = true;
        {
            std::vector<double> copy(nobs * outdim);
            auto status = umappp::initialize(neighbors, outdim, copy.data(), opt);
            status.run(copy.data());
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

TEST(Umap, SinglePrecision) {
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

    status.run(output.data());
    EXPECT_EQ(status.epoch(), 500);
    for (auto o : output){ 
        // Check that we don't get any weirdness.
        EXPECT_FALSE(std::isnan(o));
    }
}

static umappp::NeighborList<int, double> mock_neighbors(int n, int k) {
    std::mt19937_64 rng(n * 13 + k);

    umappp::NeighborList<int, double> output(n);
    std::vector<int> sampled(k);
    for (int i = 0; i < n; ++i) {
        aarand::sample(n, k, sampled.data(), rng);

        // Forcibly connecting it to the previous observation, or wrapping around to the last observation.
        // This ensures that we only have 1 component in the graph.
        const int forced = (i == 0 ? n : i) - 1; 
        bool found = false;
        for (const auto o : sampled) {
            if (o == forced) {
                found = true;
                break;
            }
        }
        if (!found) {
            sampled[0] = forced;
        }

        aarand::shuffle(sampled.begin(), sampled.size(), rng);
        double dist = 0;
        for (const int o : sampled) {
            dist += aarand::standard_uniform(rng);
            output[i].emplace_back(o, dist);
        }
    }

    return output;
}


TEST(Umap, InitializationSpectralOk) {
    int nobs = 87;
    int k = 5;

    const auto nnres = mock_neighbors(nobs, k);
    std::vector<double> ref(nobs * 2);
    umappp::initialize(nnres, 2, ref.data(), umappp::Options());
    for (auto o : ref) {
        EXPECT_NE(o, 0);
    }

    // Spectral is okay so switching fallback has no effect.
    {
        std::vector<double> output(nobs * 2);
        umappp::initialize(nnres, 2, output.data(), [&]{
            umappp::Options opt;
            opt.initialize_random_on_spectral_fail = false;
            return opt;
        }());
        EXPECT_EQ(ref, output);
    }

    // Trying with randoms.
    {
        std::vector<double> output(nobs * 2);
        auto status = umappp::initialize(nnres, 2, output.data(), [&]{
            umappp::Options opt;
            opt.initialize_method = umappp::InitializeMethod::RANDOM;
            return opt;
        }());
        for (auto o : output) {
            EXPECT_NE(o, 0);
        }
        EXPECT_NE(ref, output);
    }

    // Trying with pre-existing inputs.
    {
        std::vector<double> output(nobs * 2);
        auto status = umappp::initialize(nnres, 2, output.data(), [&]{
            umappp::Options opt;
            opt.initialize_method = umappp::InitializeMethod::NONE;
            return opt;
        }());
        for (auto o : output) {
            EXPECT_EQ(o, 0);
        }
        EXPECT_NE(ref, output);
    }
}

TEST(Umap, InitializationSpectralFail) {
    int nobs1 = 40;
    int nobs2 = 50;
    int nobs = nobs1 + nobs2;
    int k = 5;
    auto nnres = mock_neighbors(nobs1, k);
    const auto nnres2 = mock_neighbors(nobs2, k);

    // Combining the components.
    for (const auto& nn : nnres2) {
        nnres.push_back(nn);
        for (auto& n : nnres.back()) {
            n.first += nobs1; // adjusting the neighbor index.
        }
    }

    // Default fallback is random.
    {
        std::vector<double> ref(nobs * 2);
        umappp::initialize(nnres, 2, ref.data(), umappp::Options());
        for (auto o : ref) {
            EXPECT_NE(o, 0);
        }

        std::vector<double> output(nobs * 2);
        auto status = umappp::initialize(nnres, 2, output.data(), [&]{
            umappp::Options opt;
            opt.initialize_method = umappp::InitializeMethod::RANDOM;
            return opt;
        }());
        EXPECT_EQ(ref, output);
    }

    // Fallback to pre-existing inputs.
    {
        std::vector<double> ref(nobs * 2);
        umappp::initialize(nnres, 2, ref.data(), [&]{
            umappp::Options opt;
            opt.initialize_random_on_spectral_fail = false;
            return opt;
        }());

        std::vector<double> output(nobs * 2);
        auto status = umappp::initialize(nnres, 2, output.data(), [&]{
            umappp::Options opt;
            opt.initialize_method = umappp::InitializeMethod::NONE;
            return opt;
        }());
        for (auto o : output) {
            EXPECT_EQ(o, 0);
        }
        EXPECT_EQ(ref, output);
    }
}

TEST(Umap, EpochDecay) {
    EXPECT_EQ(umappp::choose_num_epochs({}, 1000), 500);
    EXPECT_LT(umappp::choose_num_epochs({}, 20000), 500);
    EXPECT_EQ(umappp::choose_num_epochs({}, 10000000), 201);
    EXPECT_EQ(umappp::choose_num_epochs({ 1000 }, 1000), 1000);
    EXPECT_EQ(umappp::choose_num_epochs({ 1000 }, 20000), 1000);
}
