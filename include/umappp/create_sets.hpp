#ifndef UMAPPP_CREATE_SETS_HPP
#define UMAPPP_CREATE_SETS_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace umappp {

template<class Searched>
void neighbor_weights(
    Searched& x, 
    int max_iter = 64, 
    double local_connectivity = 1.0, 
    double bandwidth = 1.0,
    double tol = 1e-5, 
    double min_k_dist_scale = 1e-3
) {
    double grand_mean_dist = -1;
    constexpr double max_val = std::numeric_limits<double>::max();

    #pragma omp parallel
    {
        std::vector<double> non_zero_distances;
        
        #pragma omp for
        for (size_t i = 0; i < x.size(); ++i) {
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
            const double interpolation = local_connectivity - index;
            const double lower = (index > 0 ? non_zero_distances[index - 1] : 0); // 'index' is 1-based, so -1.
            const double upper = non_zero_distances[index];
            const double rho = lower + interpolation * (upper - lower);

            // Iterating to find a good sigma, just like how t-SNE does so for beta.
            double sigma = 1.0;
            double lo = 0.0;
            double hi = max_val;
            double sigma_best = sigma;
            double adiff_min = max_val;
            const double target = std::log2(all_neighbors.size());

            bool converged = false;
            for (int iter = 0; iter < max_iter; ++iter) {
                // If distance = 0, then max(distance - rho, 0) = 0 as rho >=
                // 0. In which case, exp(-dist / sigma) is just 1 for each
                // distance of zero, allowing us to just add these directly.
                double val = n_neighbors - non_zero_distances.size();
                
                for (auto d : non_zero_distances) {
                    if (d > rho) {
                        val += std::exp(-(d - rho)/ sigma);
                    } else {
                        val += 1;
                    }
                }

                double adiff = std::abs(val - target);
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
                    sigma = (lo + hi) / 2;
                } else {
                    lo = sigma;
                    if (hi == max_val) {
                        sigma *= 2;
                    } else {
                        sigma = (lo + hi) / 2;
                    }
                }
            }

            if (!converged) {
                sigma = sigma_best;
            }

            // Quickly summing over the non-zero distances, then dividing
            // by the total number of neighbors to obtain the mean.
            double mean_dist = std::accumulate(non_zero_distances.begin(), non_zero_distances.end(), 0.0)/n_neighbors;
            sigma = std::max(min_k_dist_scale * mean_dist, sigma);

            for (int k = 0; k < n_neighbors; ++k) {
                double& dist = all_neighbors[k].second;
                if (dist > rho) {
                    dist = std::exp(-(dist - rho) / (sigma * bandwidth));
                } else {
                    dist = 1;
                }
            }
        }
    }

    return;
}

template<class Searched>
void combine_neighbor_sets(Searched& x, double mix_ratio = 1) {
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
        // neighbor's neighbors. Assuming that everything in 'searched' is
        // sorted, this should only require a single pass through the entire
        // set of neighbors as we do not need to search previously searched
        // hits.
        for (auto& y : current) {
            auto& target = x[y.first];
            auto& curlast = last[y.first];
            const auto& limits = original[y.first];
            while (curlast < limits && target[curlast].first < desired) {
                ++curlast;
            }

            if (curlast < limits && target[curlast].first == desired) {
                if (desired < y.first) { // don't average it twice.
                    double product = y.second * target[curlast].second;
                    double prob_final;

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
}

}

#endif
