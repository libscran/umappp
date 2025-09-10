#ifndef UMAPPP_COMBINE_NEIGHBOR_SETS_HPP
#define UMAPPP_COMBINE_NEIGHBOR_SETS_HPP

#include <vector>
#include <algorithm>
#include <type_traits>

#include "sanisizer/sanisizer.hpp"

#include "NeighborList.hpp"
#include "utils.hpp"

namespace umappp {

namespace internal {

template<typename Index_, typename Float_>
void combine_neighbor_sets(NeighborList<Index_, Float_>& x, const Float_ mix_ratio) {
    const Index_ num_obs = x.size(); // assume that Index_ is large enough to store the number of observations.
    auto last = sanisizer::create<std::vector<Index_> >(num_obs);
    auto original = sanisizer::create<std::vector<Index_> >(num_obs);

    for (Index_ i = 0; i < num_obs; ++i) {
        auto& current = x[i];
        std::sort(current.begin(), current.end()); // sorting by ID, see below.
        original[i] = x[i].size();
    }

    for (Index_ i = 0; i < num_obs; ++i) {
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
                    const Float_ product = y.second * target[curlast].second;
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
                    y.second = 0; // mark for deletion once the loop is done; we can't do it here as we would invalid 'last' and 'original'. 
                } else {
                    y.second *= mix_ratio;
                    target.emplace_back(i, y.second);
                }
            }
        }
    }

    // Removing zero probabilities.
    if (mix_ratio == 0) {
        for (Index_ i = 0; i < num_obs; ++i) {
            auto& current = x[i];
            auto current_size = current.size();
            decltype(I(current_size)) counter = 0;
            for (decltype(I(current_size)) j = 0; j < current_size; ++j) {
                if (current[j].second != 0) {
                    if (counter != j) {
                        current[counter] = current[j];
                    }
                    ++counter;
                }
            }
            current.resize(counter);
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
