#ifndef UMAPPP_NEIGHBOR_SIMILARITIES_HPP
#define UMAPPP_NEIGHBOR_SIMILARITIES_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "NeighborList.hpp"
#include "parallelize.hpp"

namespace umappp {

namespace internal {

/**
 * The aim of this function is to convert distances into probability-like
 * similarities using a Gaussian kernel. Our aim is to find 'sigma' such that:
 *
 * sum( exp( - max(0, dist_i - rho) / sigma ) ) = target
 *
 * Where 'rho' and 'target' are constants, and the sum is computed over all
 * neighbors 'i' for each observation.
 *
 * Note that we only need to explicitly compute the sum over neighbors where
 * 'dist > rho'. For closer neighbors, the exp() expression is equal to 1, so
 * we just add the number of such neighbors to the sum.
 *
 * We use Newton's method with a fallback to a binary search if the former
 * doesn't give sensible steps. 
 * 
 * NOTE: the UMAPPP_R_PACKAGE_TESTING macro recapitulates the gaussian kernel
 * calculation of the uwot package so that we can get a more precise comparison
 * to a trusted reference implementation. It should not be used in production. 
 */

template<typename Float_>
struct NeighborSimilaritiesOptions {
    Float_ local_connectivity = 1.0;
    Float_ bandwidth = 1.0;
    Float_ min_k_dist_scale = 1e-3; // this is only exposed for easier unit testing.
    int num_threads = 1;
};

template<bool use_newton_ = 
#ifndef UMAPPP_R_PACKAGE_TESTING
true
#else
false
#endif
, typename Index_, typename Float_>
void neighbor_similarities(NeighborList<Index_, Float_>& x, const NeighborSimilaritiesOptions<Float_>& options) {
    size_t npoints = x.size();
    const int raw_connect_index = std::floor(options.local_connectivity);
    const Float_ interpolation = options.local_connectivity - raw_connect_index;

    parallelize(options.num_threads, npoints, [&](int, size_t start, size_t length) {
        std::vector<Float_> active_delta;

        for (size_t i = start, end = start + length; i < end; ++i) {
            auto& all_neighbors = x[i];
            const int num_neighbors = all_neighbors.size();
            if (num_neighbors == 0) {
                continue;
            }

            // Define rho as the distance to the nearest (non-identical) neighbor at 'local_connectivity', possibly with interpolation.
            int num_zero = 0;
            for (const auto& f : all_neighbors) {
                if (f.second) {
                    break;
                }
                ++num_zero;
            }

            const int connect_index = num_zero + raw_connect_index;
            if (num_neighbors <= connect_index) {
                // When this happens, 'rho' is just theoretically set to the
                // maximum distance. In such cases, the weights are always just
                // set to 1 in the remaining code, because no distance can be
                // greater than 'rho'. If that's the case, we might as well
                // save some time and compute it here.
                for (int k = 0; k < num_neighbors; ++k) {
                    all_neighbors[k].second = 1;
                }
                continue;
            }

            const Float_ lower = (connect_index > 0 ? all_neighbors[connect_index - 1].second : 0); // 'local_connectivity' (and thus 'connect_index') is 1-based, hence the -1.
            const Float_ upper = all_neighbors[connect_index].second;
            const Float_ rho = lower + interpolation * (upper - lower);

            // Pre-computing the difference between each distance and rho to reduce work in the inner iterations.
            active_delta.clear();
            Float_ num_le_rho = num_zero;
            for (int k = num_zero; k < num_neighbors; ++k) {
                auto curdist = all_neighbors[k].second;
                if (curdist > rho) {
                    active_delta.push_back(curdist - rho);
                } else {
                    ++num_le_rho;
                }
            }

            if (active_delta.empty()) {
                // Same early-return logic as above.
                for (int k = 0; k < num_neighbors; ++k) {
                    all_neighbors[k].second = 1;
                }
                continue;
            }

            // Our initial sigma is chosen to match the scale of the largest delta so that we start in the right ballpark.
            Float_ sigma = 
#ifndef UMAPPP_R_PACKAGE_TESTING
                active_delta.back();
#else
                1.0
#endif
            ;

            Float_ lo = 0.0;
            constexpr Float_ max_val = std::numeric_limits<Float_>::max();
            Float_ hi = max_val;

            const Float_ target = std::log2(num_neighbors + 1) * options.bandwidth; // Based on code in uwot:::smooth_knn_matrix(). Adding 1 to include self.

            constexpr int max_iter = 64;
            for (int iter = 0; iter < max_iter; ++iter) {
                Float_ observed = num_le_rho;
                Float_ deriv = 0;

                // No need to protect against sigma = 0 as it's impossible due
                // to the bounded nature of the Newton calculation and the
                // underflow-safe nature of the binary search.
                const Float_ invsigma = 1 / sigma, invsigma2 = invsigma * invsigma;
                for (auto d : active_delta) {
                    Float_ current = std::exp(- d * invsigma);
                    observed += current;
                    deriv += d * current * invsigma2;
                }

                const Float_ diff = observed - target;
                constexpr Float_ tol = 1e-5;
                if (std::abs(diff) < tol) {
                    break;
                }

                // Refining the search interval for a (potential) binary search
                // later. We know that this function is increasing with respect
                // to increasing 'sigma', so if the diff is positive, the
                // current 'sigma' must be on the right of the root.
                if (diff > 0) {
                    hi = sigma;
                } else {
                    lo = sigma;
                }

                bool nr_ok = false;
                if constexpr(use_newton_) {
                    // Attempt a Newton-Raphson search first.
                    if (deriv) {
                        const Float_ alt_sigma = sigma - (diff / deriv); // if it overflows, we should get Inf or -Inf, so the following comparison should be fine.
                        if (alt_sigma > lo && alt_sigma < hi) {
                            sigma = alt_sigma;
                            nr_ok = true;
                        }
                    }
                }

                if (!nr_ok) {
                    // Falling back to a binary search, if Newton's method failed or was not requested.
                    if (diff > 0) {
                        sigma += (lo - sigma) / 2; // underflow-safe midpoint with the lower boundary.
                    } else {
                        if (hi == max_val) {
                            sigma *= 2;
                        } else {
                            sigma += (hi - sigma) / 2; // overflow-safe midpoint with the upper boundary.
                        }
                    }
                }
            }

            // Protect against an overly small sigma.
            Float_ mean_dist = 0;
            for (const auto& x : all_neighbors) {
                mean_dist += x.second;
            }
            mean_dist /= num_neighbors;
            sigma = std::max(options.min_k_dist_scale * mean_dist, sigma);

            const Float_ invsigma = 1 / sigma;
            for (int k = 0; k < num_neighbors; ++k) {
                Float_& dist = all_neighbors[k].second;
                if (dist > rho) {
                    dist = std::exp(-(dist - rho) * invsigma);
                } else {
                    dist = 1;
                }
            }
        }
    });

    return;
}

}

}

#endif
