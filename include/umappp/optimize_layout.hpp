#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

#include "NeighborList.hpp"
#include "aarand/aarand.hpp"

namespace umappp {

template<typename Float>
struct EpochData {
    EpochData(size_t nobs) : head(nobs) {}

    int total_epochs;
    int current_epoch = 0;

    std::vector<size_t> head;
    std::vector<int> tail;
    std::vector<Float> epochs_per_sample;

    std::vector<Float> epoch_of_next_sample;
    std::vector<Float> epoch_of_next_negative_sample;
    Float negative_sample_rate;
};

template<typename Float>
EpochData<Float> similarities_to_epochs(const NeighborList<Float>& p, int num_epochs, Float negative_sample_rate) {
    Float maxed = 0;
    size_t count = 0;
    for (const auto& x : p) {
        count += x.size();
        for (const auto& y : x) {
            maxed = std::max(maxed, y.second);
        }
    }

    EpochData<Float> output(p.size());
    output.total_epochs = num_epochs;
    output.tail.reserve(count);
    output.epochs_per_sample.reserve(count);
    const Float limit = maxed / num_epochs;

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

template<typename Float>
Float quick_squared_distance(const Float* left, const Float* right, int ndim) {
    Float dist2 = 0;
    for (int d = 0; d < ndim; ++d, ++left, ++right) {
        dist2 += (*left - *right) * (*left - *right);
    }
    constexpr Float dist_eps = std::numeric_limits<Float>::epsilon();
    return std::max(dist_eps, dist2);
}

template<typename Float>
Float clamp(Float input) {
    constexpr Float min_gradient = -4;
    constexpr Float max_gradient = 4;
    return std::min(std::max(input, min_gradient), max_gradient);
}

template<bool batch, typename Float, class Setup, class Rng> 
void optimize_sample(
    size_t i,
    int ndim,
    Float* embedding,
    Float* buffer,
    Setup& setup,
    Float a,
    Float b,
    Float gamma,
    Float alpha,
    Rng& rng,
    Float epoch
) {
    const auto& head = setup.head;
    const auto& tail = setup.tail;
    const auto& epochs_per_sample = setup.epochs_per_sample;
    auto& epoch_of_next_sample = setup.epoch_of_next_sample;
    auto& epoch_of_next_negative_sample = setup.epoch_of_next_negative_sample;
   
}

template<typename Float, class Setup, class Rng>
void optimize_layout(
    int ndim,
    Float* embedding, 
    Setup& setup,
    Float a, 
    Float b, 
    Float gamma,
    Float initial_alpha,
    Rng& rng,
    int epoch_limit
) {
    auto& n = setup.current_epoch;
    auto num_epochs = setup.total_epochs;
    auto limit_epochs = num_epochs;
    if (epoch_limit> 0) {
        limit_epochs = std::min(epoch_limit, num_epochs);
    }
    
    const size_t num_obs = setup.head.size(); 
    for (; n < limit_epochs; ++n) {
        const Float epoch = n;
        const Float alpha = initial_alpha * (1.0 - epoch / num_epochs);

        for (size_t i = 0; i < num_obs; ++i) {
            size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
            Float* left = embedding + i * ndim;

            for (size_t j = start; j < end; ++j) {
                if (setup.epoch_of_next_sample[j] > epoch) {
                    continue;
                }

                Float* right = embedding + setup.tail[j] * ndim;
                Float dist2 = quick_squared_distance(left, right, ndim);
                const Float pd2b = std::pow(dist2, b);
                const Float grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));
                {
                    Float* lcopy = left;
                    Float* rcopy = right;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                        Float gradient = alpha * clamp(grad_coef * (*lcopy - *rcopy));
                        *lcopy += gradient;
                        *rcopy -= gradient;
                    }
                }

                // Remember that 'epochs_per_negative_sample' is defined as 'epochs_per_sample[j] / negative_sample_rate'.
                // We just use it inline below rather than defining a new variable and suffering floating-point round-off.
                const size_t num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) * 
                    setup.negative_sample_rate / setup.epochs_per_sample[j]; // i.e., 1/epochs_per_negative_sample.

                for (size_t p = 0; p < num_neg_samples; ++p) {
                    size_t sampled = p % num_obs;
                    if (sampled == i) {
                        continue;
                    }

                    Float* right = embedding + sampled * ndim;
                    Float dist2 = quick_squared_distance(left, right, ndim);
                    const Float grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));

                    Float* lcopy = left;
                    const Float* rcopy = right;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++rcopy) {
                        *lcopy += alpha * clamp(grad_coef * (*lcopy - *rcopy));
                    }
                }

                setup.epoch_of_next_sample[j] += setup.epochs_per_sample[j];

                // The update to 'epoch_of_next_negative_sample' involves adding
                // 'num_neg_samples * epochs_per_negative_sample', which eventually boils
                // down to setting epoch_of_next_negative_sample to 'epoch'.
                setup.epoch_of_next_negative_sample[j] = epoch;
            }
        }
    }

    return;
}

}

#endif
