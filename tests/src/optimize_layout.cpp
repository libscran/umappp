#include <gtest/gtest.h>

#include "umappp/neighbor_similarities.hpp"
#include "umappp/combine_neighbor_sets.hpp"
#include "umappp/optimize_layout.hpp"
#include <vector>
#include "knncolle/knncolle.hpp"

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
        umappp::combine_neighbor_sets(stored, 1);
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

    auto epoch = umappp::similarities_to_epochs(stored, 500);
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

INSTANTIATE_TEST_SUITE_P(
    Optimize,
    OptimizeTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors
    )
);
