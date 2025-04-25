#ifndef UMAPPP_COMBINE_NEIGHBOR_SETS_HPP
#define UMAPPP_COMBINE_NEIGHBOR_SETS_HPP

#include <vector>
#include <algorithm>
#include <type_traits>

#include "NeighborList.hpp"

namespace umappp {

namespace internal {

template<typename Index_, typename Float_>
void combine_neighbor_sets(NeighborList<Index_, Float_>& x, Float_ mix_ratio) {
    Index_ num_obs = x.size();
    std::vector<decltype(x[0].size())> last(num_obs), original(num_obs);

    for (Index_ i = 0; i < num_obs; ++i) {
        auto& current = x[i];
        std::sort(current.begin(), current.end()); // sorting by ID, see below.
        original[i] = x[i].size();
    }

    for (Index_ i = 0; i < num_obs; ++i ){
        auto& current = x[i];

        // Looping through the neighbors and searching for self in each
        // neighbor's neighbors. As each inner vector in 'x' is sorted,
        // this should only require a single pass through the entire set of
        // neighbors as we do not need to search previously searched hits.
        for (auto& y : current) {
            auto& target = x[y.first];
            auto& curlast = last[y.first];
            const auto& limits = original[y.first];
            while (curlast < limits && target[curlast].first < i) {
                ++curlast;
            }

            if (curlast < limits && target[curlast].first == i) {
                // If i > y.first, then this would have already been done in a
                // previous iteration of the outermost loop where i and y.first
                // swap values. So we skip this to avoid adding it twice.
                if (i < y.first) { 
                    Float_ product = y.second * target[curlast].second;
                    Float_ prob_final;

                    if (mix_ratio == 1) {
                        prob_final = y.second + target[curlast].second - product;
                    } else if (mix_ratio == 0) {
                        prob_final = product;
                    } else {
                        prob_final = mix_ratio * (y.second + target[curlast].second - product) + (1 - mix_ratio) * product;
                    }

                    y.second = prob_final;
                    target[curlast].second = prob_final;
                }
            } else {
                if (mix_ratio == 1) {
                    target.emplace_back(i, y.second);
                } else if (mix_ratio == 0) {
                    y.second = 0; // mark for deletion.
                } else {
                    y.second *= mix_ratio;
                    target.emplace_back(i, y.second);
                }
            }
        }
    }

    // Removing zero probabilities.
    if (mix_ratio == 0) {
        for (auto& current : x) {
            typename std::remove_reference<decltype(current)>::type replacement;
            replacement.reserve(current.size());
            for (const auto& y : current) {
                if (y.second) {
                    replacement.push_back(y);
                }
            }
            std::swap(current, replacement);
        }
    }

    // Sorting everything by index to be more cache-friendly. Also,
    // irlba::ParallelSparseMatrix needs increasing inserts.
    for (auto& current : x) {
        std::sort(current.begin(), current.end());
    }

    return;
}

}

}

#endif
