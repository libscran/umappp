#include <gtest/gtest.h>

#include "umappp/create_sets.hpp"
#include "knncolle/knncolle.hpp"

#include <map>

class SetCreationTest : public ::testing::TestWithParam<std::tuple<int, int> > {
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

TEST_P(SetCreationTest, NeighborWeights) {
    assemble(GetParam());
    auto stored = generate_neighbors();
    
    umappp::neighbor_weights(stored);

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

TEST_P(SetCreationTest, Clones) {
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
    umappp::neighbor_weights(stored);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(stored[i][0].second, 1);
        EXPECT_EQ(stored[i][1].second, 1);
    }
}

TEST_P(SetCreationTest, EdgeCases) {
    assemble(GetParam());
    auto stored = generate_neighbors();

    // Forcing the fallback when local_connectivity is too high.
    umappp::neighbor_weights(stored, 64, 100); 

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

    umappp::neighbor_weights(stored);
    auto ref = stored[0];
    for (int i = 0; i < k; ++i) {
        for (auto j : stored[i]) {
            EXPECT_EQ(j.second, 1);
        }
    }
}

template<class Searched>
void check_symmetry(const Searched& x) {
    std::map<std::pair<int, int>, double> probs;

    for (size_t i = 0; i < x.size(); ++i) {
        const auto& y = x[i];
        for (const auto& z : y) {
            EXPECT_TRUE(z.second > 0); // double-check that probabilities are positive.

            std::pair<int, int> target;
            target.first = std::max(z.first, static_cast<int>(i));
            target.second = std::min(z.first, static_cast<int>(i));

            auto it = probs.find(target);
            if (it != probs.end()) {
                EXPECT_EQ(it->second, z.second); 
                probs.erase(it);
            } else {
                probs[target] = z.second;
            }
        }
    }

    // if it's symmetric, all element should be lost.
    EXPECT_EQ(probs.size(), 0);
}

TEST_P(SetCreationTest, Combining) {
    assemble(GetParam());
    auto stored = generate_neighbors();
    umappp::neighbor_weights(stored);

    auto Union = stored;
    umappp::combine_neighbor_sets(Union, 1);
    check_symmetry(Union);

    auto intersect = stored;
    umappp::combine_neighbor_sets(intersect, 0);
    check_symmetry(intersect);

    auto middle = stored;
    umappp::combine_neighbor_sets(middle, 0.5);
    check_symmetry(middle);

    // Comparing the number of edges.
    size_t total_u = 0, total_i = 0, total_o = 0;
    for (size_t i = 0; i < stored.size(); ++i) {
        const auto& uvec = Union[i];
        const auto& ivec = intersect[i];
        const auto& mvec = middle[i];
        const auto& orig = stored[i];
        
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
    SetCreation,
    SetCreationTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors 
    )
);
