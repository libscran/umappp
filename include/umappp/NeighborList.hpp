#ifndef UMAPPP_NEIGHBOR_LIST_HPP
#define UMAPPP_NEIGHBOR_LIST_HPP

/**
 * @file NeighborList.hpp
 *
 * @brief Defines the `NeighborList` typedef.
 */

namespace umappp {

/**
 * @brief Neighbor specification based on index and distance.
 *
 * @tparam Float Floating-point type.
 *
 * The index refers to the position of the neighboring observation in the dataset.
 * The statistic can store some statistic related to the neighbor, e.g., distance or probabilities.
 */ 
template<typename Float = double>
using Neighbor =  std::pair<int, Float>;

/**
 * @brief Lists of neighbors for each observation.
 *
 * @tparam Float Floating-point type.
 *
 * Each inner vector corresponds to an observation and contains the list of nearest neighbors for that observation.
 */
template<typename Float = double>
using NeighborList = std::vector<std::vector<Neighbor<Float> > >;

}

#endif
