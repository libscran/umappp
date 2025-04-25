#include <gtest/gtest.h>

#include "umappp/combine_neighbor_sets.hpp"
#include "knncolle/knncolle.hpp"

#include <map>
#include <random>
#include <cmath>

class CombineNeighborSetTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    std::vector<std::vector<std::pair<int, double> > > neighbors;

    void SetUp() { 
        auto p = GetParam();
        size_t nobs = std::get<0>(p);
        int k = std::get<1>(p);
        int ndim = 5;

        std::mt19937_64 rng(nobs * k); // for some variety
        std::normal_distribution<> dist(0, 1);

        std::vector<double> data(nobs * ndim);
        for (size_t r = 0; r < data.size(); ++r) {
            data[r] = dist(rng);
        }

        auto builder = knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::EuclideanDistance<double, double> >());
        auto index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
        auto searcher = index->initialize();

        neighbors.resize(nobs);
        std::vector<int> indices;
        std::vector<double> distances;
        for (size_t i = 0; i < nobs; ++i) {
            searcher->search(i, k, &indices, &distances);

            size_t actual_k = indices.size();
            for (size_t x = 0; x < actual_k; ++x) {
                neighbors[i].emplace_back(indices[x], std::exp(-distances[x]));
            }
        }
    }

protected:
    template<class Searched_>
    static void slow_symmetrization(const Searched_& original, const Searched_& combined, double mix_ratio = 1) {
        std::map<std::pair<int, int>, double> probs;

        // Filling 'probs'.
        for (size_t i = 0; i < original.size(); ++i) {
            const auto& y = original[i];
            for (const auto& z : y) {
                std::pair<int, int> target;
                target.first = std::max(z.first, static_cast<int>(i));
                target.second = std::min(z.first, static_cast<int>(i));

                auto it = probs.find(target);
                if (it != probs.end()) {
                    it->second *= -1;
                    if (mix_ratio == 1) {
                        it->second += z.second - z.second * it->second;
                    } else if (mix_ratio == 0) {
                        it->second *= z.second;
                    } else {
                        double prod = it->second * z.second;
                        it->second = mix_ratio * (z.second + it->second - prod) + (1 - mix_ratio) * prod;
                    }
                } else {
                    probs[target] = -z.second;
                }
            }
        }

        auto it = probs.begin();
        while (it != probs.end()) {
            auto& x = *it;
            if (x.second < 0) {
                if (mix_ratio == 0) {
                    probs.erase(it++);
                    continue;
                }
                
                if (mix_ratio == 1) {
                    x.second *= -1;
                } else {
                    x.second *= -mix_ratio;
                }
            }
            ++it;
        }

        // Comparing to the combined results.
        std::map<std::pair<int, int>, bool> found;
        for (size_t i = 0; i < combined.size(); ++i) {
            const auto& y = combined[i];
            for (const auto& z : y) {
                std::pair<int, int> target;
                target.first = std::max(z.first, static_cast<int>(i));
                target.second = std::min(z.first, static_cast<int>(i));

                auto it = probs.find(target);
                EXPECT_TRUE(it != probs.end());
                if (it != probs.end()) {
                    EXPECT_FLOAT_EQ(it->second, z.second);
                }
                found[target] = true;
            }
        }

        EXPECT_EQ(probs.size(), found.size());
    }
};

TEST_P(CombineNeighborSetTest, Combining) {
    auto Union = neighbors;
    umappp::internal::combine_neighbor_sets<>(Union, 1.0);
    slow_symmetrization(neighbors, Union, 1);

    auto intersect = neighbors;
    umappp::internal::combine_neighbor_sets<>(intersect, 0.0);
    slow_symmetrization(neighbors, intersect, 0);

    auto middle = neighbors;
    umappp::internal::combine_neighbor_sets<>(middle, 0.5);
    slow_symmetrization(neighbors, middle, 0.5);

    // Comparing the number of edges.
    size_t total_u = 0, total_i = 0, total_o = 0;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto& uvec = Union[i];
        const auto& ivec = intersect[i];
        const auto& mvec = middle[i];
        const auto& orig = neighbors[i];
        
        EXPECT_TRUE(uvec.size() == mvec.size());
        EXPECT_TRUE(uvec.size() >= ivec.size());
        EXPECT_TRUE(uvec.size() >= orig.size());
        EXPECT_TRUE(ivec.size() <= orig.size());

        total_u += uvec.size();
        total_i += ivec.size();
        total_o += orig.size();
    }

    EXPECT_TRUE(total_u > total_i);
    EXPECT_TRUE(total_u > total_o);
    EXPECT_TRUE(total_i < total_o);
}

INSTANTIATE_TEST_SUITE_P(
    CombineNeighborSet,
    CombineNeighborSetTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors 
    )
);
