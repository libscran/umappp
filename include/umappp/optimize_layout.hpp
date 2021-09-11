#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "NeighborList.hpp"
#include "aarand/aarand.hpp"

#ifdef _OPENMP
#include "omp.h"
#endif

namespace umappp {

struct EpochData {
    EpochData(size_t nobs) : head(nobs) {}

    int total_epochs;
    int current_epoch = 0;

    std::vector<size_t> head;
    std::vector<int> tail;
    std::vector<double> epochs_per_sample;

    std::vector<double> epoch_of_next_sample;
    std::vector<double> epoch_of_next_negative_sample;
    double negative_sample_rate;
};

inline EpochData similarities_to_epochs(const NeighborList& p, int num_epochs, double negative_sample_rate) {
    double maxed = 0;
    size_t count = 0;
    for (const auto& x : p) {
        count += x.size();
        for (const auto& y : x) {
            maxed = std::max(maxed, y.second);
        }
    }

    EpochData output(p.size());
    output.total_epochs = num_epochs;
    output.tail.reserve(count);
    output.epochs_per_sample.reserve(count);
    const double limit = maxed / num_epochs;

    size_t last = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const auto& x = p[i];
        for (const auto& y : x) {
            if (y.second >= limit) {
                output.tail.push_back(y.first);
                output.epochs_per_sample.push_back(maxed / y.second);
                ++last;
            }
        }
        output.head[i] = last;
    }

    // Filling in some epoch-related running statistics.
    output.epoch_of_next_sample = output.epochs_per_sample;
    output.epoch_of_next_negative_sample = output.epochs_per_sample;
    for (auto& e : output.epoch_of_next_negative_sample) {
        e /= negative_sample_rate;
    }
    output.negative_sample_rate = negative_sample_rate;

    return output;       
}

template<class Setup, class Rng>
inline void optimize_layout(
    int ndim,
    double* embedding, 
    Setup& setup,
    double a, 
    double b, 
    double gamma,
    double initial_alpha,
    Rng& rng,
    int epoch_limit
) {
    auto& n = setup.current_epoch;
    auto num_epochs = setup.total_epochs;
    auto limit_epochs = num_epochs;
    if (limit_epochs > 0) {
        limit_epochs = std::min(limit_epochs, num_epochs);
    }

    auto& head = setup.head;
    auto& tail = setup.tail;
    auto& epochs_per_sample = setup.epochs_per_sample;
    auto& epoch_of_next_sample = setup.epoch_of_next_sample;
    auto& epoch_of_next_negative_sample = setup.epoch_of_next_negative_sample;

    const size_t num_obs = head.size(); 
    constexpr double dist_eps = std::numeric_limits<double>::epsilon();
    const double negative_sample_rate = setup.negative_sample_rate;

    // Defining the gradient parameters.
    constexpr double min_gradient = -4;
    constexpr double max_gradient = 4;

    for (; n < limit_epochs; ++n) {
        const double alpha = initial_alpha * (1.0 - static_cast<double>(n) / num_epochs);

        for (size_t i = 0; i < setup.head.size(); ++i) {
            size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
            double* left = embedding + i * ndim;

            for (size_t j = start; j < end; ++j) {
                if (epoch_of_next_sample[j] > n) {
                    continue;
                }

                double dist2 = 0;
                double* right = embedding + setup.tail[j] * ndim;
                {
                    const double* lcopy = left;
                    const double* rcopy = right;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                        dist2 += (*lcopy - *rcopy) * (*lcopy - *rcopy);
                    }
                    dist2 = std::max(dist_eps, dist2);
                }

                const double pd2b = std::pow(dist2, b);
                const double grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));
                {
                    double* lcopy = left;
                    double* rcopy = right;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                        double gradient = alpha * std::min(std::max(grad_coef * (*lcopy - *rcopy), min_gradient), max_gradient);
                        *lcopy += gradient;
                        *rcopy -= gradient;
                    }
                }

                // Here, we divide by epochs_per_negative_sample, defined as epochs_per_sample[j] / negative_sample_rate.
                const size_t num_neg_samples = (n - epoch_of_next_negative_sample[j]) * negative_sample_rate / epochs_per_sample[j];

                for (size_t p = 0; p < num_neg_samples; ++p) {
                    size_t sampled = aarand::discrete_uniform(rng, num_obs); 
                    if (sampled == i) {
                        continue;
                    }

                    double dist2 = 0;
                    double* right = embedding + sampled * ndim;
                    {
                        const double* lcopy = left;
                        const double* rcopy = right;
                        for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                            dist2 += (*lcopy - *rcopy) * (*lcopy - *rcopy);
                        }
                        dist2 = std::max(dist_eps, dist2);
                    }

                    const double grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));
                    {
                        double* lcopy = left;
                        const double* rcopy = right;
                        for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                            *lcopy += alpha * std::min(std::max(grad_coef * (*lcopy - *rcopy), min_gradient), max_gradient);
                        }
                    }
                }

                epoch_of_next_sample[j] += epochs_per_sample[j];

                // The update involves adding num_neg_samples * epoch_of_next_negative_sample[j],
                // which eventually boils down to setting this to 'n'.
                epoch_of_next_negative_sample[j] = n;
            }
        }
    }

    return;
}

#ifdef _OPENMP
template<class Setup, class SeedFunction, class EngineFunction>
inline void optimize_layout_batched(
    int ndim,
    double* embedding, 
    Setup& setup,
    double a, 
    double b, 
    double gamma,
    double initial_alpha,
    SeedFunction seeder,
    EngineFunction creator,
    int epoch_limit
) {
    auto& n = setup.current_epoch;
    auto num_epochs = setup.total_epochs;
    auto limit_epochs = num_epochs;
    if (limit_epochs > 0) {
        limit_epochs = std::min(limit_epochs, num_epochs);
    }

    auto& head = setup.head;
    auto& tail = setup.tail;
    auto& epochs_per_sample = setup.epochs_per_sample;
    auto& epoch_of_next_sample = setup.epoch_of_next_sample;
    auto& epoch_of_next_negative_sample = setup.epoch_of_next_negative_sample;

    const size_t num_obs = head.size(); 
    constexpr double dist_eps = std::numeric_limits<double>::epsilon();
    const double negative_sample_rate = setup.negative_sample_rate;

    // Defining the gradient parameters.
    constexpr double min_gradient = -4;
    constexpr double max_gradient = 4;

    std::vector<decltype(seeder())> seeds(num_obs);
    std::vector<double> replace_buffer(num_obs * ndim);
    double* replacement = replace_buffer.data();
    bool using_replacement = false;

    for (; n < limit_epochs; ++n) {
        const double alpha = initial_alpha * (1.0 - static_cast<double>(n) / num_epochs);

        // Fill the seeds.
        for (auto& s : seeds) {
            s = seeder();
        }

        // Input and outputs alternate between epochs, to avoid the need for a
        // copy operation on the entire embedding at the end of each epoch.
        const double* reference = (using_replacement ? replacement : embedding); 
        double* output = (using_replacement ? embedding : replacement);
        using_replacement = !using_replacement;

        #pragma omp parallel for
        for (size_t i = 0; i < setup.head.size(); ++i) {
            auto localrng = creator(seeds[i]);
            size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];

            // Making a local copy to avoid false sharing.
            const double* left = reference + i * ndim;
            std::vector<double> buffer(left, left + ndim);

            for (size_t j = start; j < end; ++j) {
                if (epoch_of_next_sample[j] > n) {
                    continue;
                }

                double dist2 = 0;
                const double* right = reference + setup.tail[j] * ndim;
                {
                    const double* lcopy = left;
                    const double* rcopy = right;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                        dist2 += (*lcopy - *rcopy) * (*lcopy - *rcopy);
                    }
                    dist2 = std::max(dist_eps, dist2);
                }

                const double pd2b = std::pow(dist2, b);
                const double grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));
                {
                    double* lcopy = buffer.data();
                    const double* rcopy = right;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                        double gradient = alpha * std::min(std::max(grad_coef * (*lcopy - *rcopy), min_gradient), max_gradient);

                        // Doubling as we'll assume symmetry from the same
                        // force applied by the right node. This allows us to
                        // avoid worrying about accounting for modifications to
                        // the right node.
                        *lcopy += 2 * gradient; 
                    }
                }

                const size_t num_neg_samples = (n - epoch_of_next_negative_sample[j]) * negative_sample_rate / epochs_per_sample[j];
                for (size_t p = 0; p < num_neg_samples; ++p) {
                    size_t sampled = aarand::discrete_uniform(localrng, num_obs); 
                    if (sampled == i) {
                        continue;
                    }

                    double dist2 = 0;
                    const double* right = reference + sampled * ndim;
                    {
                        const double* lcopy = left;
                        const double* rcopy = right;
                        for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                            dist2 += (*lcopy - *rcopy) * (*lcopy - *rcopy);
                        }
                        dist2 = std::max(dist_eps, dist2);
                    }

                    const double grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));
                    {
                        double* lcopy = buffer.data();
                        const double* rcopy = right;
                        for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                            *lcopy += alpha * std::min(std::max(grad_coef * (*lcopy - *rcopy), min_gradient), max_gradient);
                        }
                    }
                }

                epoch_of_next_sample[j] += epochs_per_sample[j];
                epoch_of_next_negative_sample[j] = n;
            }

            std::copy(buffer.begin(), buffer.end(), output + i * ndim);
        }
    }

    if (using_replacement) {
        std::copy(replace_buffer.begin(), replace_buffer.end(), embedding);
    }

    return;
}
#endif

}

#endif
