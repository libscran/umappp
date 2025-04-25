#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#ifndef UMAPPP_NO_PARALLEL_OPTIMIZATION
#include <thread>
#include <atomic>
#else
#include <stdexcept>
#endif

#include "NeighborList.hpp"
#include "aarand/aarand.hpp"

namespace umappp {

namespace internal {

template<typename Index_, typename Float_>
struct EpochData {
    EpochData(Index_ nobs) : head(nobs) {}

    int total_epochs;
    int current_epoch = 0;

    std::vector<std::size_t> head;
    std::vector<Index_> tail;
    std::vector<Float_> epochs_per_sample;

    std::vector<Float_> epoch_of_next_sample;
    std::vector<Float_> epoch_of_next_negative_sample;
    Float_ negative_sample_rate;
};

template<typename Index_, typename Float_>
EpochData<Index_, Float_> similarities_to_epochs(const NeighborList<Index_, Float_>& p, int num_epochs, Float_ negative_sample_rate) {
    Float_ maxed = 0;
    std::size_t count = 0;
    for (const auto& x : p) {
        count += x.size();
        for (const auto& y : x) {
            maxed = std::max(maxed, y.second);
        }
    }

    EpochData<Index_, Float_> output(p.size());
    output.total_epochs = num_epochs;
    output.tail.reserve(count);
    output.epochs_per_sample.reserve(count);
    const Float_ limit = maxed / num_epochs;

    std::size_t last = 0;
    for (Index_ i = 0, num_obs = p.size(); i < num_obs; ++i) {
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

template<typename Float_>
Float_ quick_squared_distance(const Float_* left, const Float_* right, std::size_t num_dim) {
    Float_ dist2 = 0;
    for (std::size_t d = 0; d < num_dim; ++d) {
        Float_ delta = (left[d] - right[d]);
        dist2 += delta * delta;
    }
    constexpr Float_ dist_eps = std::numeric_limits<Float_>::epsilon();
    return std::max(dist_eps, dist2);
}

template<typename Float_>
Float_ clamp(Float_ input) {
    constexpr Float_ min_gradient = -4;
    constexpr Float_ max_gradient = 4;
    return std::min(std::max(input, min_gradient), max_gradient);
}

template<typename Index_, typename Float_>
unsigned long long compute_num_neg_samples(std::size_t j, Float_ epoch, const EpochData<Index_, Float_>& setup) {
    // Remember that 'epochs_per_negative_sample' is defined as 'epochs_per_sample[j] / negative_sample_rate'.
    // We just use it inline below rather than defining a new variable and suffering floating-point round-off.
    Float_ num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) * 
        setup.negative_sample_rate / setup.epochs_per_sample[j]; // i.e., 1/epochs_per_negative_sample.

    // Avoiding problems with overflow. We return an unsigned long long to guarantee at least 64 bits,
    // which should be more than enough to hold whatever num_neg_samples is.
    constexpr auto max_value = std::numeric_limits<unsigned long long>::max();
    if (num_neg_samples <= static_cast<Float_>(max_value)) {
        return num_neg_samples;
    } else {
        return max_value;
    }
}

/*****************************************************
 ***************** Serial code ***********************
 *****************************************************/

template<typename Index_, typename Float_, class Rng_>
void optimize_layout(
    std::size_t num_dim,
    Float_* embedding, 
    EpochData<Index_, Float_>& setup,
    Float_ a, 
    Float_ b, 
    Float_ gamma,
    Float_ initial_alpha,
    Rng_& rng,
    int epoch_limit
) {
    auto& n = setup.current_epoch;
    auto num_epochs = setup.total_epochs;
    auto limit_epochs = num_epochs;
    if (epoch_limit> 0) {
        limit_epochs = std::min(epoch_limit, num_epochs);
    }

    for (; n < limit_epochs; ++n) {
        const Float_ epoch = n;
        const Float_ alpha = initial_alpha * (1.0 - epoch / num_epochs);

        Index_ num_obs = setup.head.size(); 
        for (Index_ i = 0; i < num_obs; ++i) {
            std::size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
            Float_* left = embedding + static_cast<std::size_t>(i) * num_dim; // cast to size_t to avoid overflow.

            for (std::size_t j = start; j < end; ++j) {
                if (setup.epoch_of_next_sample[j] > epoch) {
                    continue;
                }

                {
                    Float_* right = embedding + static_cast<std::size_t>(setup.tail[j]) * num_dim; // again, casting to avoid overflow.
                    Float_ dist2 = quick_squared_distance(left, right, num_dim);
                    const Float_ pd2b = std::pow(dist2, b);
                    const Float_ grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));

                    for (std::size_t d = 0; d < num_dim; ++d) {
                        auto& l = left[d];
                        auto& r = right[d];
                        Float_ gradient = alpha * clamp(grad_coef * (l - r));
                        l += gradient;
                        r -= gradient;
                    }
                }

                auto num_neg_samples = compute_num_neg_samples(j, epoch, setup);
                for (decltype(num_neg_samples) p = 0; p < num_neg_samples; ++p) {
                    auto sampled = aarand::discrete_uniform(rng, num_obs);
                    if (sampled == i) {
                        continue;
                    }

                    const Float_* right = embedding + static_cast<std::size_t>(sampled) * num_dim; // again, casting to avoid overflow.
                    Float_ dist2 = quick_squared_distance(left, right, num_dim);
                    const Float_ grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));

                    for (std::size_t d = 0; d < num_dim; ++d) {
                        left[d] += alpha * clamp(grad_coef * (left[d] - right[d]));
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

/*****************************************************
 **************** Parallel code **********************
 *****************************************************/

#ifndef UMAPPP_NO_PARALLEL_OPTIMIZATION
template<typename Index_, typename Float_>
class BusyWaiterThread {
private:
    // Settable parameters.
    std::vector<Index_> my_selections;
    std::vector<unsigned char> my_skips;
    Index_ my_observation;
    Float_ my_alpha;

    // Private constructor parameters. 
    std::size_t my_num_dim;
    Index_ my_sentinel;
    Float_* my_embedding;
    const EpochData<Index_, Float_>* my_setup;
    Float_ my_a;
    Float_ my_b;
    Float_ my_gamma;

    // Private internal parameters.
    std::vector<Float_> my_self_modified;
    std::thread my_worker;
    std::atomic<bool> my_ready = false;
    bool my_finished = false;
    bool my_active = false;

public:
    void run() {
        my_ready.store(true, std::memory_order_release);
    }

    void wait() {
        while (my_ready.load(std::memory_order_acquire)) {
            ;
        }
    }

    std::vector<Index_>& get_selections() {
        return my_selections;
    }

    std::vector<unsigned char>& get_skips() {
        return my_skips;
    }

    Index_& get_observation() {
        return my_observation;
    }

    Float_& get_alpha() {
        return my_alpha;
    }

    void migrate_parameters(BusyWaiterThread& src) {
        my_selections.swap(src.my_selections);
        my_skips.swap(src.my_skips);
        my_alpha = src.my_alpha;
        my_observation = src.my_observation;
    }

    void transfer_coordinates() {
        std::size_t offset = static_cast<std::size_t>(my_observation) * my_num_dim; // cast to avoid overflow.
        std::copy(my_self_modified.begin(), my_self_modified.end(), my_embedding + offset);
    }

public:
    void run_direct() {
        auto seIt = my_selections.begin();
        auto skIt = my_skips.begin();
        const std::size_t start = (my_observation == 0 ? 0 : my_setup->head[my_observation-1]), end = my_setup->head[my_observation];

        // Copying it over into a thread-local buffer to avoid false sharing.
        // We don't bother doing this for the neighbors, though, as it's 
        // tedious to make sure that the modified values are available during negative sampling.
        // (This isn't a problem for the self, as the self cannot be its own negative sample.)
        {
            const Float_* left = my_embedding + static_cast<std::size_t>(my_observation) * my_num_dim; // cast to avoid overflow.
            std::copy_n(left, my_num_dim, my_self_modified.data());
        }

        for (std::size_t j = start; j < end; ++j) {
            if (*(skIt++)) {
                continue;
            }

            {
                Float_* left = my_self_modified.data();
                Float_* right = my_embedding + static_cast<std::size_t>(my_setup->tail[j]) * my_num_dim; // cast to avoid overflow.

                Float_ dist2 = quick_squared_distance(left, right, my_num_dim);
                const Float_ pd2b = std::pow(dist2, my_b);
                const Float_ grad_coef = (-2 * my_a * my_b * pd2b) / (dist2 * (my_a * pd2b + 1.0));

                for (std::size_t d = 0; d < my_num_dim; ++d) {
                    auto& l = left[d];
                    auto& r = right[d];
                    Float_ gradient = my_alpha * clamp(grad_coef * (l - r));
                    l += gradient;
                    r -= gradient;
                }
            }

            while (seIt != my_selections.end() && *seIt != my_sentinel) {
                Float_* left = my_self_modified.data();
                const Float_* right = my_embedding + static_cast<std::size_t>(*seIt) * my_num_dim; // cast to avoid overflow.

                Float_ dist2 = quick_squared_distance(left, right, my_num_dim);
                const Float_ grad_coef = 2 * my_gamma * my_b / ((0.001 + dist2) * (my_a * std::pow(dist2, my_b) + 1.0));

                for (std::size_t d = 0; d < my_num_dim; ++d) {
                    left[d] += my_alpha * clamp(grad_coef * (left[d] - right[d]));
                }
                ++seIt;
            }
            ++seIt; // get past the sentinel.
        }
    }

public:
    BusyWaiterThread(std::size_t num_dim, Index_ sentinel, Float_* embedding, const EpochData<Index_, Float_>* setup, Float_ a, Float_ b, Float_ gamma) : 
        my_num_dim(num_dim),
        my_sentinel(sentinel),
        my_embedding(embedding),
        my_setup(setup),
        my_a(a), 
        my_b(b),
        my_gamma(gamma),
        my_self_modified(my_num_dim)
    {}

    void start() {
        my_active = true;
        my_worker = std::thread(&BusyWaiterThread::loop, this);
    }
 
private:
    void loop() {
        while (true) {
            while (!my_ready.load(std::memory_order_acquire)) {
                ;
            }
            if (my_finished) {
                break;
            }
            run_direct();
            my_ready.store(false, std::memory_order_release);
        }
    }

public:
    ~BusyWaiterThread() {
        if (my_active) {
            my_finished = true;
            my_ready.store(true, std::memory_order_release);
            my_worker.join();
        }
    }

    // Note that the atomic is not moveable, so this move constructor strictly
    // mimics the regular constructor. As such, this had better not be called
    // after start() (i.e., when the worker thread is running).
    BusyWaiterThread(BusyWaiterThread&& src) :
        my_num_dim(src.my_num_dim),
        my_embedding(src.my_embedding),
        my_setup(src.my_setup),
        my_a(src.my_a), 
        my_b(src.my_b),
        my_gamma(src.my_gamma),
        my_self_modified(std::move(src.my_self_modified))
    {}

    // Ditto the comment above.
    BusyWaiterThread& operator=(BusyWaiterThread&& src) {
        my_num_dim = src.my_num_dim;
        my_embedding = src.my_embedding;
        my_setup = src.my_setup;
        my_a = src.my_a;
        my_b = src.my_b;
        my_gamma = src.my_gamma;
        my_self_modified = std::move(src.my_self_modified);
        return *this;
    }

    BusyWaiterThread& operator=(const BusyWaiterThread&) = delete;
    BusyWaiterThread(const BusyWaiterThread& src) = delete;
};
#endif

//#define PRINT false

template<typename Index_, typename Float_, class Rng_>
void optimize_layout_parallel(
    std::size_t num_dim,
    Float_* embedding, 
    EpochData<Index_, Float_>& setup,
    Float_ a, 
    Float_ b, 
    Float_ gamma,
    Float_ initial_alpha,
    Rng_& rng,
    int epoch_limit,
    int nthreads
) {
#ifndef UMAPPP_NO_PARALLEL_OPTIMIZATION
    auto& n = setup.current_epoch;
    auto num_epochs = setup.total_epochs;
    auto limit_epochs = num_epochs;
    if (epoch_limit> 0) {
        limit_epochs = std::min(epoch_limit, num_epochs);
    }

    const Index_ num_obs = setup.head.size(); 
    const auto sentinel = num_obs; // This is a valid sentinel as no negative-sampled index should have value equal to the number of observations.

    // We run some things directly in this main thread to avoid excessive busy-waiting.
    BusyWaiterThread<Index_, Float_> staging(num_dim, sentinel, embedding, &setup, a, b, gamma);

    int nthreadsm1 = nthreads - 1;
    std::vector<BusyWaiterThread<Index_, Float_> > pool;
    pool.reserve(nthreadsm1);
    for (int t = 0; t < nthreadsm1; ++t) {
        pool.emplace_back(num_dim, sentinel, embedding, &setup, a, b, gamma);
        pool.back().start();
    }

    std::vector<Index_> last_touched_iteration(num_obs);
    std::vector<unsigned char> touch_type(num_obs);
    std::vector<int> threads_in_progress;

    for (; n < limit_epochs; ++n) {
        const Float_ epoch = n;
        const Float_ alpha = initial_alpha * (1.0 - epoch / num_epochs);

        // Iteration is 1-based so as to allow last_touched_iteration[i] = 0 to
        // mean that it has never been touched. A touch after the first
        // iteration is instead last_touched_iteration[i] = 1. (We can't use
        // last_touched_iteration[i] = -1 because Index_ might be unsigned.)
        Index_ base_iteration = 1;
        std::fill(last_touched_iteration.begin(), last_touched_iteration.end(), 0);

        Index_ i = 0;
        while (i < num_obs) {
            bool is_clear = true;
//            if (PRINT) { std::cout << "size is " << threads_in_progress.size() << std::endl; }

            for (int t = threads_in_progress.size(); t < nthreads && i < num_obs; ++t) {
                staging.get_alpha() = alpha;
                staging.get_observation() = i;

                // Tapping the RNG here in the serial section.
                auto& selections = staging.get_selections();
                selections.clear();
                auto& skips = staging.get_skips();
                skips.clear();

                const Index_ self_iteration = i + 1; // remember, iterations are 1-based.
                constexpr unsigned char READONLY = 0;
                constexpr unsigned char WRITE = 1;

                {
                    auto& touched = last_touched_iteration[i];
                    auto& ttype = touch_type[i];
//                    if (PRINT) { std::cout << "SELF: " << i << ": " << touched << " (" << ttype << ")" << std::endl; }
                    if (touched >= base_iteration) {
                        is_clear = false;
//                        if (PRINT) { std::cout << "=== FAILED! ===" << std::endl; }
                    }
                    touched = self_iteration;
                    ttype = WRITE;
                }

                const std::size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
                for (std::size_t j = start; j < end; ++j) {
                    bool skip = setup.epoch_of_next_sample[j] > epoch;
                    skips.push_back(skip);
                    if (skip) {
                        continue;
                    }

                    {
                        auto neighbor = setup.tail[j];
                        auto& touched = last_touched_iteration[neighbor];
                        auto& ttype = touch_type[neighbor];
//                        if (PRINT) { std::cout << "\tNEIGHBOR: " << neighbor << ": " << touched << " (" << ttype << ")" << std::endl; }
                        if (touched >= base_iteration) {
                            if (touched != self_iteration) {
                                is_clear = false;
//                                if (PRINT) { std::cout << "=== FAILED! ===" << std::endl; }
                            }
                        }
                        touched = self_iteration;
                        ttype = WRITE;
                    }

                    auto num_neg_samples = compute_num_neg_samples(j, epoch, setup);
                    for (decltype(num_neg_samples) p = 0; p < num_neg_samples; ++p) {
                        Index_ sampled = aarand::discrete_uniform(rng, num_obs);
                        if (sampled == i) {
                            continue;
                        }
                        selections.push_back(sampled);

                        auto& touched = last_touched_iteration[sampled];
                        auto& ttype = touch_type[sampled];
//                        if (PRINT) { std::cout << "\t\tSAMPLED: " << sampled << ": " << touched << " (" << ttype << ")" << std::endl; }
                        if (touched >= base_iteration) { 
                            if (touched != self_iteration) {
                                if (ttype == WRITE) {
                                    is_clear = false;
//                                    if (PRINT) { std::cout << "=== FAILED! ===" << std::endl; }
                                }
                            }
                        } else {
                            // Only updating if it wasn't touched by a previous thread in this
                            // round of thread iterations.
                            ttype = READONLY;
                            touched = self_iteration;
                        }
                    }

                    selections.push_back(sentinel);

                    setup.epoch_of_next_sample[j] += setup.epochs_per_sample[j];
                    setup.epoch_of_next_negative_sample[j] = epoch;
                }

                if (!is_clear) {
                    // As we only updated the access for 'sampled' to READONLY
                    // if they weren't touched by another thread, we need to go
                    // through and manually update them now that the next round
                    // of thread_iterations will use 'self_iteration' as the
                    // 'base_iteration'. This ensures that the flags are properly
                    // set for the next round, under the expectation that the
                    // pending thread becomes the first thread.
                    for (auto s : selections) {
                        if (s != sentinel) {
                            auto& touched = last_touched_iteration[s];
                            if (touched != self_iteration) {
                                touched = self_iteration;
                                touch_type[s] = READONLY;
                            }
                        }
                    }
                    break;
                } 

                // Submitting if it's not the final job, otherwise just running it directly.
                // This avoids a busy-wait on the main thread that uses up an extra CPU.
                if (t < nthreadsm1) {
                    const int thread_index = i % nthreadsm1;
                    pool[thread_index].migrate_parameters(staging);
                    pool[thread_index].run();
                    threads_in_progress.push_back(thread_index);
                } else {
                    staging.run_direct();
                    staging.transfer_coordinates();
                }

                ++i;
            }

            // Waiting for all the jobs that were submitted.
            for (auto thread : threads_in_progress) {
                pool[thread].wait();
                pool[thread].transfer_coordinates();
            }
            threads_in_progress.clear();

//            if (PRINT) { std::cout << "###################### OK ##########################" << std::endl; }

            base_iteration = i + 1; // remember, iterations are 1-based.
            if (!is_clear) {
                const int thread_index = i % nthreadsm1;
                pool[thread_index].migrate_parameters(staging);
                pool[thread_index].run();
                threads_in_progress.push_back(thread_index);
                ++i;
            }
        }

        for (auto thread : threads_in_progress) {
            pool[thread].wait();
            pool[thread].transfer_coordinates();
        }
        threads_in_progress.clear();
    }

    return;
#else
    throw std::runtime_error("umappp was not compiled with support for parallel optimization");
#endif
}

}

}

#endif
