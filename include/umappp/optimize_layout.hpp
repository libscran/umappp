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
struct Lock {
    Lock() {
        omp_init_nest_lock(&lock_);
    }
    ~Lock() {
        omp_destroy_nest_lock(&lock_);
    }
    void lock() {
        omp_set_nest_lock(&lock_);
    }
    void unlock() {
        omp_unset_nest_lock(&lock_);
    }
private:
    omp_nest_lock_t lock_;
};

template<class Setup, class Rng>
inline void optimize_layout_parallel(
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

    Lock glock;
    std::vector<Lock> olocks(num_obs * omp_get_max_threads());

    for (; n < limit_epochs; ++n) {
        const double alpha = initial_alpha * (1.0 - static_cast<double>(n) / num_epochs);
        size_t i_ = 0;

        #pragma omp parallel
        {
            std::vector<std::pair<size_t, size_t> > tails;
            std::vector<size_t> random_draws;
            std::vector<double> head(ndim);
            double * left = head.data();
            int nthreads = omp_get_num_threads();
            int curthread = omp_get_thread_num();

            while (1) {
                tails.clear();
                random_draws.clear();

                size_t i;
                { 
                    /* Serial block: each thread picks up the latest job. As
                     * only one thread can add locks at any given time, this is
                     * safe in terms of both race conditions and deadlocks. It 
                     * also ensures that random numbers are drawn in sequence.
                     */
                    glock.lock(); 
                    if (i_ == setup.head.size()) {
                        glock.unlock();
                        break;
                    }

                    i = i_;
                    ++i_;
                    {
                        size_t x = i * nthreads;
                        for (int l = 0; l < nthreads; ++l, ++x) {
                            olocks[x].lock();
                        }
                    }

                    size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
                    for (size_t j = start; j < end; ++j) {
                        if (epoch_of_next_sample[j] > n) {
                            continue;
                        }

                        auto other = setup.tail[j];
                        {
                            size_t x = other * nthreads;
                            for (int l = 0; l < nthreads; ++l, ++x) {
                                olocks[x].lock();
                            }
                        }
 
                        const size_t num_neg_samples = (n - epoch_of_next_negative_sample[j]) * negative_sample_rate / epochs_per_sample[j];
                        for (size_t p = 0; p < num_neg_samples; ++p) {
                            size_t sampled = aarand::discrete_uniform(rng, num_obs); 
                            if (sampled == i) {
                                continue;
                            }

                            random_draws.push_back(sampled);
                            olocks[sampled * nthreads + curthread].lock();
                        }

                        tails.emplace_back(other, random_draws.size());
                        epoch_of_next_sample[j] += epochs_per_sample[j];
                        epoch_of_next_negative_sample[j] = n;
                    }

                    glock.unlock(); 
                }

                // Making a local copy, avoid problems with false sharing.
                double* original_left = embedding + i * ndim;
                std::copy(original_left, original_left + ndim, left);
                size_t previous_random = 0;

                for (auto current : tails) {
                    double dist2 = 0;
                    double* right = embedding + current.first * ndim;
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

                    {
                        size_t x = current.first * nthreads;
                        for (int l = 0; l < nthreads; ++l, ++x) {
                            olocks[x].unlock();
                        }
                    }

                    while (previous_random < current.second) {
                        size_t sampled = random_draws[previous_random];
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

                        olocks[sampled * nthreads + curthread].unlock();

                        ++previous_random;
                    }
                }

                std::copy(left, left + ndim, original_left);
                {
                    size_t x = i * nthreads;
                    for (int l = 0; l < nthreads; ++l, ++x) {
                        olocks[x].unlock();
                    }
                }
            }
        }
    }

    return;
}

#endif

}

#endif
