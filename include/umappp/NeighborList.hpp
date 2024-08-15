#ifndef UMAPPP_NEIGHBOR_LIST_HPP
#define UMAPPP_NEIGHBOR_LIST_HPP

#include <utility>
#include <vector>

/**
 * @file NeighborList.hpp
 *
 * @brief Defines the `NeighborList` alias.
 */

namespace umappp {

/**
 * @brief Lists of neighbors for each observation.
 *
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Float_ Floating-point type for the distances.
 *
 * Each inner vector corresponds to an observation and contains the list of nearest neighbors for that observation, sorted by increasing distance.
 * Neighbors for each observation should be unique - there should be no more than one occurrence of each index in each inner vector.
 * Also, the inner vector for observation `i` should not contain any `Neighbor` with index `i`.
 */
template<typename Index_, typename Float_>
using NeighborList = std::vector<std::vector<std::pair<Index_, Float_> > >;

}

#endif
