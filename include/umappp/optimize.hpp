#ifndef UMAPPP_OPTIMIZE_HPP
#define UMAPPP_OPTIMIZE_HPP

#include <random>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace umappp {

template<class Prob, class Gradient>
void optimize_layout(
    int num_dim,
    size_t num_obs, 
    double* embedding, 
    const Prob& probabilities, 
    int num_epochs,
    const std::vector<double>& epochs_per_sample,
    double a, 
    double b, 
    double gamma,
    double initial_alpha, 
    double negative_sample_rate
) {
    constexpr double dist_eps = std::numeric_limits<double>::epsilon();
    std::vector<uint_fast64_t> seeds(num_obs);
    std::mt19937_64 overlord(123456790);

    // Defining epoch-related constants.
    std::vector<double> epochs_per_next_sample(epochs_per_sample); 
    std::vector<double> epochs_per_negative_sample(epochs_per_sample);
    for (auto& e : epochs_per_negative_sample) {
        e /= negative_sample_rate;
    }
    std::vector<double> epochs_per_next_negative_sample(epochs_per_negative_sample);

    // Defining the gradient parameters.
    constexpr double min_gradient = -4;
    constexpr double max_gradient = 4;

    for (int n = 0; n < num_epochs; ++n) {
        const double alpha = initial_alpha * (1.0 - static_cast<double>(n) / n_epochs);

        // Generating seeds for each observation. This ensures that we get the
        // same results regardless of the number of threads.
        for (auto& x : seeds) {
            x = overlord();
        }

        #pragma omp parallel
        {
            #pragma omp for
            for (size_t i = 0; i < num_obs; ++i) {
                if (epoch_of_next_sample[i] > n) {
                    continue;
                }
                std::mt19937_64 rng(seeds[i]);

                for (const auto& neighbor : probabilities[i]) {
                    double dist2 = 0;
                    {
                        const double* left = embedding + i * num_dim;
                        const double* right = embedding.data() + neighbor.first * num_dim;
                        for (int d = 0; d < num_dim; ++d, ++left, ++right) {
                            dist2 += (*left - *right) * (*left - *right);
                        }
                        ist2 = std::max(dist_eps, dist2);
                    }

                    const double pd2b = std::pow(dist2, b);
                    const double grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));
                    {
                        double* left = embedding + i * num_dim;
                        double* right = embedding + neighbor.first * num_dim;
                        for (int d = 0; d < num_dim; ++d, ++left, ++right) {
                            double gradient = alpha * std::min(std::max(grad_coef * (*left - *right), min_gradient), max_gradient);
                            *left += gradient;
                            *right -= gradient;
                        }
                    }

                    const size_t num_neg_samples = (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i];
                    for (size_t p = 0; p < num_neg_samples; ++p) {
                        size_t sampled = (rng() % num_obs) * num_dim; // TODO: fix the sampler
                        if (sampled == i) {
                            continue;
                        }

                        double dist2 = 0;
                        {
                            const double* left = embedding + i * num_dim;
                            const double* right = embedding + sampled * num_dim;
                            for (int d = 0; d < num_dim; ++d, ++left, ++right) {
                                dist2 += (*left - *right) * (*left - *right);
                            }
                            dist2 = std::max(dist_eps, dist2);
                        }

                        const double grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));
                        {
                            double* left = embedding + i * num_dim;
                            double* right = embedding + sampled * num_dim;
                            for (int d = 0; d < num_dim; ++d, ++left, ++right) {
                                *left += alpha * std::min(std::max(grad_coef * (*left - *right), min_gradient), max_gradient);
                            }
                        }
                    }

                    epoch_of_next_sample[i] += epochs_per_sample[i];
                    epoch_of_next_negative_sample[i] += num_neg_samples * epochs_per_negative_sample[i];
                }
            }
        }
    }

    return;
}

}

#endif
