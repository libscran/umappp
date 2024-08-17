#ifndef UMAPPP_COMBINE_NEIGHBOR_SETS_HPP
#define UMAPPP_COMBINE_NEIGHBOR_SETS_HPP

#include <vector>
#include <algorithm>
#include <type_traits>

#include "NeighborList.hpp"

namespace umappp {

namespace internal {

template<typename Index_, typename Float_>
void combine_neighbor_sets(NeighborList<Index_, Float_>& x, Float_ mix_ratio = 1) {
    std::vector<size_t> last(x.size());
    std::vector<size_t> original(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        auto& current = x[i];
        std::sort(current.begin(), current.end()); // sorting by ID, see below.
        original[i] = x[i].size();
    }

    for (size_t first = 0; first < x.size(); ++first) {
        auto& current = x[first];
        const int desired = first;

        // Looping through the neighbors and searching for self in each
        // neighbor's neighbors. As each inner vector in 'x' is sorted,
        // this should only require a single pass through the entire set of
        // neighbors as we do not need to search previously searched hits.
        for (auto& y : current) {
            auto& target = x[y.first];
            auto& curlast = last[y.first];
            const auto& limits = original[y.first];
            while (curlast < limits && target[curlast].first < desired) {
                ++curlast;
            }

            if (curlast < limits && target[curlast].first == desired) {
                if (desired < y.first) { // don't average it twice.
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
                    target.emplace_back(desired, y.second);
                } else if (mix_ratio == 0) {
                    y.second = 0; // mark for deletion.
                } else {
                    y.second *= mix_ratio;
                    target.emplace_back(desired, y.second);
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
