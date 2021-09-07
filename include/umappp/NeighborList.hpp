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
 * The index refers to the position of the neighboring observation in the dataset.
 * The statistic is used to store the distance to the neighbor.
 */ 
typedef std::pair<int, double> Neighbor;

/**
 * @brief Lists of neighbors for each observation.
 *
 * Each inner vector corresponds to an observation and contains the list of nearest neighbors for that observation.
 */ 
typedef std::vector<std::vector<Neighbor> > NeighborList;

}

#endif
