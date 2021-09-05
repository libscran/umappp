#include <gtest/gtest.h>

#include "umappp/spectral_init.hpp"

#include <random>
#include <vector>

void symmetrize(umappp::NeighborList& x) {
    std::vector<size_t> available;
    available.reserve(x.size());
    for (const auto& y : x) {
        available.push_back(y.size());
    }

    for (size_t i = 0; i < x.size(); ++i) {
        auto& current = x[i];
        for (size_t j = 0; j < available[i]; ++j) {
            x[current[j].first].emplace_back(i, current[j].second);
        }
    }

    return;
}


TEST(ComponentTest, Simple) {
    int order = 5;

    umappp::NeighborList edges(order);
    edges[4].emplace_back(0, 0.5);
    edges[4].emplace_back(1, 0.5);
    edges[3].emplace_back(2, 0.5);

    auto copy = edges;
    symmetrize(copy);
    auto components = umappp::find_components(copy);

    EXPECT_EQ(components.ncomponents(), 2);
    std::vector<int>expected{ 0, 0, 1, 1, 0 };
    EXPECT_EQ(components.components, expected);

    // Merging into one component.
    edges[3].emplace_back(1, 0.5);

    copy = edges;
    symmetrize(copy);
    components = umappp::find_components(copy);

    EXPECT_EQ(components.ncomponents(), 1);
    expected =std::vector<int>{ 0, 0, 0, 0, 0 };
    EXPECT_EQ(components.components, expected);
}

TEST(ComponentTest, Singletons) {
    int order = 5;

    umappp::NeighborList edges(order);
    auto components = umappp::find_components(edges);

    EXPECT_EQ(components.ncomponents(), 5);
    std::vector<int>expected { 0, 1, 2, 3, 4 };
    EXPECT_EQ(components.components, expected);

    // Sticking in an edge to merge nodes.
    edges[3].emplace_back(1, 0.5);

    auto copy = edges;
    symmetrize(copy);
    components = umappp::find_components(copy);

    EXPECT_EQ(components.ncomponents(), 4);
    expected = std::vector<int>{ 0, 1, 2, 1, 3 };
    EXPECT_EQ(components.components, expected);
}

TEST(ComponentTest, Ordering) {
    int order = 6;

    // Deliberately checking the case where one node splits into two,
    // or two nodes merge into one, depending on whether we are 
    // traversing in order of increasing or decreasing index.
    umappp::NeighborList edges(order);
    edges[4].emplace_back(2, 0.5);
    edges[4].emplace_back(3, 0.5);

    edges[5].emplace_back(1, 0.5);
    edges[5].emplace_back(0, 0.5);

    symmetrize(edges);
    auto components = umappp::find_components(edges);
    EXPECT_EQ(components.ncomponents(), 2);
    std::vector<int>expected{ 0, 0, 1, 1, 1, 0 };
    EXPECT_EQ(components.components, expected);
}
