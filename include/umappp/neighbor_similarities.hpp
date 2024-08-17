#ifndef UMAPPP_NEIGHBOR_SIMILARITIES_HPP
#define UMAPPP_NEIGHBOR_SIMILARITIES_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "NeighborList.hpp"

namespace umappp {

namespace internal {

template<typename Index_, typename Float_>
void neighbor_similarities(
    NeighborList<Index_, Float_>& x, 
    Float_ local_connectivity = 1.0,
    Float_ bandwidth = 1.0, 
    int max_iter = 64, 
    Float_ tol = 1e-5, 
    Float_ min_k_dist_scale = 1e-3,
    [[maybe_unused]] int num_threads = 1
) {
    constexpr Float_ max_val = std::numeric_limits<Float_>::max();
    size_t npoints = x.size();

#ifndef UMAPPP_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads)
#endif
    {
#else
    UMAPPP_CUSTOM_PARALLEL(npoints, [&](size_t first, size_t last) -> void {
#endif

        std::vector<Float_> non_zero_distances;

#ifndef UMAPPP_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp for
#endif
        for (size_t i = 0; i < npoints; ++i) {
#else
        for (size_t i = first; i < last; ++i) {
#endif

            auto& all_neighbors = x[i];
            const int n_neighbors = all_neighbors.size();

            non_zero_distances.clear();
            for (const auto& f : all_neighbors) {
                if (f.second) {
                    non_zero_distances.push_back(f.second);
                }
            }

            if (non_zero_distances.size() <= local_connectivity) {
                // When this happens, 'rho' is just theoretically set to the
                // maximum distance. In such cases, the weights are always just
                // set to 1 in the remaining code, because no distance can be
                // greater than 'rho'. If that's the case, we might as well
                // save some time and compute it here.
                for (int k = 0; k < n_neighbors; ++k) {
                    all_neighbors[k].second = 1;
                }
                continue;
            }

            // Find rho, the distance to the nearest (non-identical) neighbor,
            // possibly with interpolation.
            int index = std::floor(local_connectivity);
            const Float_ interpolation = local_connectivity - index;
            const Float_ lower = (index > 0 ? non_zero_distances[index - 1] : 0); // 'index' is 1-based, so -1.
            const Float_ upper = non_zero_distances[index];
            const Float_ rho = lower + interpolation * (upper - lower);

            // Iterating to find a good sigma, just like how t-SNE does so for beta.
            Float_ sigma = 1.0;
            Float_ lo = 0.0;
            Float_ hi = max_val;
            Float_ sigma_best = sigma;
            Float_ adiff_min = max_val;
            const Float_ target = std::log2(all_neighbors.size() + 1); // +1 to include self. Dunno why, but uwot does it.

            bool converged = false;
            for (int iter = 0; iter < max_iter; ++iter) {
                // If distance = 0, then max(distance - rho, 0) = 0 as rho >=
                // 0. In which case, exp(-dist / sigma) is just 1 for each
                // distance of zero, allowing us to just add these directly.
                Float_ val = n_neighbors - non_zero_distances.size();
                
                for (auto d : non_zero_distances) {
                    if (d > rho) {
                        val += std::exp(-(d - rho)/ sigma);
                    } else {
                        val += 1;
                    }
                }

                Float_ adiff = std::abs(val - target);
                if (adiff < tol) {
                    converged = true;
                    break;
                }

                // store best sigma in case binary search fails (usually in the presence
                // of multiple degenerate distances)
                if (adiff < adiff_min) {
                    adiff_min = adiff;
                    sigma_best = sigma;
                }

                if (val > target) {
                    hi = sigma;
                    sigma += (lo - sigma) / 2; // overflow-safe midpoint with the lower boundary.
                } else {
                    lo = sigma;
                    if (hi == max_val) {
                        sigma *= 2;
                    } else {
                        sigma += (hi - sigma) / 2; // overflow-safe midpoint with the upper boundary.
                    }
                }
            }

            if (!converged) {
                sigma = sigma_best;
            }

            // Quickly summing over the non-zero distances, then dividing
            // by the total number of neighbors to obtain the mean.
            Float_ mean_dist = std::accumulate(non_zero_distances.begin(), non_zero_distances.end(), 0.0)/n_neighbors;
            sigma = std::max(min_k_dist_scale * mean_dist, sigma);

            for (int k = 0; k < n_neighbors; ++k) {
                Float_& dist = all_neighbors[k].second;
                if (dist > rho) {
                    dist = std::exp(-(dist - rho) / (sigma * bandwidth));
                } else {
                    dist = 1;
                }
            }

#ifndef UMAPPP_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, num_threads);
#endif

    return;
}

}

}

#endif
