#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <random>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "NeighborList.hpp"

namespace umappp {

struct EpochData {
    std::vector<size_t> head;
    std::vector<int> tail;
    std::vector<double> epochs_per_sample;
    int num_epochs;
};

inline EpochData similarities_to_epochs(const NeighborList& p, int num_epochs) {
    double maxed = 0;
    size_t count = 0;
    for (const auto& x : p) {
        count += x.size();
        for (const auto& y : x) {
            maxed = std::max(maxed, y.second);
        }
    }

    EpochData output;
    output.num_epochs = num_epochs;
    output.head.resize(p.size());
    output.tail.reserve(count);
    output.epochs_per_sample.reserve(count);
    const double limit = maxed / num_epochs;

    size_t last = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const auto& x = p[i];
        for (const auto& y : x) {
            if (y.second >= limit) {
                output.tail.push_back(y.first);
                output.epochs_per_sample.push_back(y.second / limit);
                ++last;
            }
        }
        output.head[i] = last;
    }

    return output;       
}

inline void optimize_layout(
    int num_dim,
    double* embedding, 
    const EpochData& epochs,
    double a, 
    double b, 
    double gamma,
    double initial_alpha, 
    double negative_sample_rate
) {
    constexpr double dist_eps = std::numeric_limits<double>::epsilon();
    std::mt19937_64 overlord(123456790);
    const size_t num_obs = epochs.head.size(); 
    const int num_epochs = epochs.num_epochs;

    // Defining epoch-related constants.
    const std::vector<double>& epochs_per_sample = epochs.epochs_per_sample;
    std::vector<double> epoch_of_next_sample(epochs_per_sample); 
    std::vector<double> epoch_of_next_negative_sample(epochs_per_sample);
    for (auto& e : epoch_of_next_negative_sample) {
        e /= negative_sample_rate;
    }

    // Defining the gradient parameters.
    constexpr double min_gradient = -4;
    constexpr double max_gradient = 4;

    for (int n = 0; n < num_epochs; ++n) {
        const double alpha = initial_alpha * (1.0 - static_cast<double>(n) / num_epochs);

        for (size_t i = 0; i < epochs.head.size(); ++i) {
            size_t start = (i == 0 ? 0 : epochs.head[i-1]), end = epochs.head[i];
            double* left = embedding + i * num_dim;

            for (size_t j = start; j < end; ++j) {
                if (epoch_of_next_sample[j] > n) {
                    continue;
                }

                double dist2 = 0;
                double* right = embedding + epochs.tail[j] * num_dim;
                {
                    const double* lcopy = left;
                    const double* rcopy = right;
                    for (int d = 0; d < num_dim; ++d, ++lcopy, ++rcopy) {
                        dist2 += (*lcopy - *rcopy) * (*lcopy - *rcopy);
                    }
                    dist2 = std::max(dist_eps, dist2);
                }

                const double pd2b = std::pow(dist2, b);
                const double grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));
                {
                    double* lcopy = left;
                    double* rcopy = right;
                    for (int d = 0; d < num_dim; ++d, ++lcopy, ++rcopy) {
                        double gradient = alpha * std::min(std::max(grad_coef * (*lcopy - *rcopy), min_gradient), max_gradient);
                        *lcopy += gradient;
                        *rcopy -= gradient;
                    }
                }

                const double epochs_per_negative_sample = epochs_per_sample[j] / negative_sample_rate;
                const size_t num_neg_samples = (n - epoch_of_next_negative_sample[j]) / epochs_per_negative_sample;

                for (size_t p = 0; p < num_neg_samples; ++p) {
                    size_t sampled = (overlord() % num_obs) * num_dim; // TODO: fix the sampler
                    if (sampled == i) {
                        continue;
                    }

                    double dist2 = 0;
                    double* right = embedding + sampled * num_dim;
                    {
                        const double* lcopy = left;
                        const double* rcopy = right;
                        for (int d = 0; d < num_dim; ++d, ++lcopy, ++rcopy) {
                            dist2 += (*lcopy - *rcopy) * (*lcopy - *rcopy);
                        }
                        dist2 = std::max(dist_eps, dist2);
                    }

                    const double grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));
                    {
                        double* lcopy = left;
                        const double* rcopy = right;
                        for (int d = 0; d < num_dim; ++d, ++lcopy, ++rcopy) {
                            *lcopy += alpha * std::min(std::max(grad_coef * (*lcopy - *rcopy), min_gradient), max_gradient);
                        }
                    }
                }

                epoch_of_next_sample[j] += epochs_per_sample[j];
                epoch_of_next_negative_sample[j] = num_neg_samples * epochs_per_negative_sample;
            }
        }
    }

    return;
}

}

#endif
