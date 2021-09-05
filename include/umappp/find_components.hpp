#ifndef UMAPPP_FIND_COMPONENTS_HPP
#define UMAPPP_FIND_COMPONENTS_HPP

#include <vector>
#include "NeighborList.hpp"

namespace umappp {

struct ComponentIndices {
    ComponentIndices(int ncomponents, std::vector<int> comp) : components(std::move(comp)), new_indices(components.size()), reversed(ncomponents) {
        for (size_t i = 0; i < components.size(); ++i) {
            auto m = components[i];
            new_indices[i] = reversed[m].size();
            reversed[m].push_back(i);
        }
        return;
    }

    size_t ncomponents() const {
        return reversed.size();
    }

    std::vector<int> components;
    std::vector<int> new_indices;
    std::vector<std::vector<int> > reversed;
};

/* Finds the connected components of the graph. This assumes that
 * the edges are already symmetrized by combine_neighbor_sets,
 * allowing us to use a single-pass through the edge list.
 */
inline ComponentIndices find_components(const NeighborList& edges) {
    int counter = 0;
    std::vector<int> mapping(edges.size(), -1);

    for (size_t current = 0; current < edges.size(); ++current) {
        if (mapping[current] != -1) {
            continue;
        }

        std::vector<int> remaining(1, current);
        mapping[current] = counter;
        do {
            int curfriend = remaining.back();
            remaining.pop_back();

            for (const auto& ff : edges[curfriend]) {
                if (mapping[ff.first] == -1) {
                    remaining.push_back(ff.first);
                    mapping[ff.first] = counter;
                }
            }
        } while (remaining.size());

        ++counter;
    }

    return ComponentIndices(counter, std::move(mapping));
}

}

#endif
