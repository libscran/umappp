#include <gtest/gtest.h>

#include "umappp/spectral_init.hpp"

#include <random>
#include <vector>

TEST(ComponentTest, Simple) {
    int order = 5;

    umappp::Edges edges;
    edges.emplace_back(4, 0, 0.5);
    edges.emplace_back(4, 1, 0.5);
    edges.emplace_back(3, 2, 0.5);

    auto components = umappp::find_components(edges, order);
    EXPECT_EQ(components.first, 2);
    std::vector<int>expected{ 0, 0, 1, 1, 0 };
    EXPECT_EQ(components.second, expected);

    // Merging into one component.
    edges.emplace_back(3, 1, 0.5);
    components = umappp::find_components(edges, order);
    EXPECT_EQ(components.first, 1);
    expected =std::vector<int>{ 0, 0, 0, 0, 0 };
    EXPECT_EQ(components.second, expected);
}

TEST(ComponentTest, Singletons) {
    int order = 5;
    umappp::Edges edges;
    auto components = umappp::find_components(edges, order);
    EXPECT_EQ(components.first, 5);
    std::vector<int>expected { 0, 1, 2, 3, 4 };

    // Sticking in an edge to merge nodes.
    edges.emplace_back(3, 1, 0.5);
    components = umappp::find_components(edges, order);
    EXPECT_EQ(components.first, 4);
    expected = std::vector<int>{ 0, 1, 2, 1, 3 };
    EXPECT_EQ(components.second, expected);
}

TEST(ComponentTest, Ordering) {
    int order = 6;

    // Deliberately checking the case where one node splits into two,
    // or two nodes merge into one, depending on whether we are 
    // traversing in order of increasing or decreasing index.
    umappp::Edges edges;
    edges.emplace_back(4, 2, 0.5);
    edges.emplace_back(4, 3, 0.5);

    edges.emplace_back(5, 1, 0.5);
    edges.emplace_back(5, 0, 0.5);

    auto components = umappp::find_components(edges, order);
    EXPECT_EQ(components.first, 2);
    std::vector<int>expected{ 0, 1, 1, 1, 0, 0 };
    EXPECT_EQ(components.second, expected);
}
