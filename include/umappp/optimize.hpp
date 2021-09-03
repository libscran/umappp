#ifndef UMAPPP_OPTIMIZE_HPP
#define UMAPPP_OPTIMIZE_HPP

#include <random>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace umappp {

inline void optimize_layout(
    int num_dim,
    double* embedding, 
    const std::vector<size_t>& head, 
    const std::vector<int>& tail, 
    const std::vector<double>& epochs_per_sample,
    int num_epochs,
    double a, 
    double b, 
    double gamma,
    double initial_alpha, 
    double negative_sample_rate
) {
    constexpr double dist_eps = std::numeric_limits<double>::epsilon();
    std::mt19937_64 overlord(123456790);
    const size_t num_obs = head.size(); 

    // Defining epoch-related constants.
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

        for (size_t i = 0; i < head.size(); ++i) {
            size_t start = (i == 0 ? 0 : head[i-1]), end = head[i];
            double* left = embedding + i * num_dim;

            for (size_t j = start; j < end; ++j) {
                if (epoch_of_next_sample[j] > n) {
                    continue;
                }

                double dist2 = 0;
                double* right = embedding + tail[j] * num_dim;
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

                // const double epochs_per_negative_sample = epochs_per_sample[j] / negative_sample_rate;
                const size_t num_neg_samples = (n - epoch_of_next_negative_sample[j]) * (negative_sample_rate / epochs_per_sample[i]);

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
                epoch_of_next_negative_sample[j] = n; // i.e., num_neg_samples * epochs_per_negative_sample;
            }
        }
    }

    return;
}

}

#endif
