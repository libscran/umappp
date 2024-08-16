#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdint>

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

    std::vector<size_t> head;
    std::vector<Index_> tail;
    std::vector<Float_> epochs_per_sample;

    std::vector<Float_> epoch_of_next_sample;
    std::vector<Float_> epoch_of_next_negative_sample;
    Float_ negative_sample_rate;
};

template<typename Index_, typename Float_>
EpochData<Index_, Float_> similarities_to_epochs(const NeighborList<Index_, Float_>& p, int num_epochs, Float_ negative_sample_rate) {
    Float_ maxed = 0;
    size_t count = 0;
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

template<typename Float_>
Float_ quick_squared_distance(const Float_* left, const Float_* right, size_t ndim) {
    Float_ dist2 = 0;
    for (size_t d = 0; d < ndim; ++d) {
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

/*****************************************************
 ***************** Serial code ***********************
 *****************************************************/

template<typename Index_, typename Float_, class Rng_>
void optimize_layout(
    size_t ndim, // use a size_t here for safer pointer arithmetic.
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

    const size_t num_obs = setup.head.size(); 
    for (; n < limit_epochs; ++n) {
        const Float_ epoch = n;
        const Float_ alpha = initial_alpha * (1.0 - epoch / num_epochs);

        for (size_t i = 0; i < num_obs; ++i) {
            size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
            Float_* left = embedding + i * ndim; // everything is already a size_t, no need to cast.

            for (size_t j = start; j < end; ++j) {
                if (setup.epoch_of_next_sample[j] > epoch) {
                    continue;
                }

                {
                    Float_* right = embedding + static_cast<size_t>(setup.tail[j]) * ndim;
                    Float_ dist2 = quick_squared_distance(left, right, ndim);
                    const Float_ pd2b = std::pow(dist2, b);
                    const Float_ grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));

#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (size_t d = 0; d < ndim; ++d) {
                        auto& l = left[d];
                        auto& r = right[d];
                        Float_ gradient = alpha * clamp(grad_coef * (l - r));
                        l += gradient;
                        r -= gradient;
                    }
                }

                // Remember that 'epochs_per_negative_sample' is defined as 'epochs_per_sample[j] / negative_sample_rate'.
                // We just use it inline below rather than defining a new variable and suffering floating-point round-off.
                const size_t num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) * 
                    setup.negative_sample_rate / setup.epochs_per_sample[j]; // i.e., 1/epochs_per_negative_sample.

                for (size_t p = 0; p < num_neg_samples; ++p) {
                    size_t sampled = aarand::discrete_uniform(rng, num_obs);
                    if (sampled == i) {
                        continue;
                    }

                    const Float_* right = embedding + sampled * ndim; // already size_t's, no need to cast.
                    Float_ dist2 = quick_squared_distance(left, right, ndim);
                    const Float_ grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));

#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (size_t d = 0; d < ndim; ++d) {
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
    std::vector<size_t> my_selections;
    std::vector<unsigned char> my_skips;
    size_t my_observation;
    Float_ my_alpha;

    // Private constructor parameters. 
    size_t my_ndim;
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

    std::vector<size_t>& get_selections() {
        return my_selections;
    }

    std::vector<unsigned char>& get_skips() {
        return my_skips;
    }

    size_t& get_observation() {
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
        size_t offset = my_observation * my_ndim; // both already size_t's, no need to cast.
        std::copy(my_self_modified.begin(), my_self_modified.end(), my_embedding + offset);
    }

public:
    void run_direct() {
        auto seIt = my_selections.begin();
        auto skIt = my_skips.begin();
        const size_t start = (my_observation == 0 ? 0 : my_setup->head[my_observation-1]), end = my_setup->head[my_observation];

        // Copying it over into a thread-local buffer to avoid false sharing.
        // We don't bother doing this for the neighbors, though, as it's 
        // tedious to make sure that the modified values are available during negative sampling.
        // (This isn't a problem for the self, as the self cannot be its own negative sample.)
        {
            const Float_* left = my_embedding + my_observation * my_ndim;
            std::copy_n(left, my_ndim, my_self_modified.data());
        }

        for (size_t j = start; j < end; ++j) {
            if (*(skIt++)) {
                continue;
            }

            {
                Float_* left = my_self_modified.data();
                Float_* right = my_embedding + static_cast<size_t>(my_setup->tail[j]) * my_ndim; // cast to avoid overflow.

                Float_ dist2 = quick_squared_distance(left, right, my_ndim);
                const Float_ pd2b = std::pow(dist2, my_b);
                const Float_ grad_coef = (-2 * my_a * my_b * pd2b) / (dist2 * (my_a * pd2b + 1.0));

#ifdef _OPENMP
                #pragma omp simd
#endif
                for (size_t d = 0; d < my_ndim; ++d) {
                    auto& l = left[d];
                    auto& r = right[d];
                    Float_ gradient = my_alpha * clamp(grad_coef * (l - r));
                    l += gradient;
                    r -= gradient;
                }
            }

            while (seIt != my_selections.end() && *seIt != static_cast<size_t>(-1)) {
                Float_* left = my_self_modified.data();
                const Float_* right = my_embedding + (*seIt) * my_ndim;

                Float_ dist2 = quick_squared_distance(left, right, my_ndim);
                const Float_ grad_coef = 2 * my_gamma * my_b / ((0.001 + dist2) * (my_a * std::pow(dist2, my_b) + 1.0));

#ifdef _OPENMP
                #pragma omp simd
#endif
                for (size_t d = 0; d < my_ndim; ++d) {
                    left[d] += my_alpha * clamp(grad_coef * (left[d] - right[d]));
                }
                ++seIt;
            }
            ++seIt; // get past the -1.
        }
    }

public:
    BusyWaiterThread(int ndim, Float_* embedding, const EpochData<Index_, Float_>* setup, Float_ a, Float_ b, Float_ gamma) : 
        my_ndim(ndim),
        my_embedding(embedding),
        my_setup(setup),
        my_a(a), 
        my_b(b),
        my_gamma(gamma),
        my_self_modified(my_ndim)
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
        my_ndim(src.my_ndim),
        my_embedding(src.my_embedding),
        my_setup(src.my_setup),
        my_a(src.my_a), 
        my_b(src.my_b),
        my_gamma(src.my_gamma),
        my_self_modified(std::move(src.my_self_modified))
    {}

    // Ditto the comment above.
    BusyWaiterThread& operator=(BusyWaiterThread&& src) {
        my_ndim = src.my_ndim;
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
    size_t ndim, // use a size_t here for safer pointer arithmetic.
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

    const size_t num_obs = setup.head.size(); 
    std::vector<int> last_touched(num_obs);
    std::vector<unsigned char> touch_type(num_obs);

    // We run some things directly in this main thread to avoid excessive busy-waiting.
    BusyWaiterThread<Index_, Float_> staging(ndim, embedding, &setup, a, b, gamma);

    int nthreadsm1 = nthreads - 1;
    std::vector<BusyWaiterThread<Index_, Float_> > pool;
    pool.reserve(nthreadsm1);
    for (int t = 0; t < nthreadsm1; ++t) {
        pool.emplace_back(ndim, embedding, &setup, a, b, gamma);
        pool.back().start();
    }

    std::vector<int> jobs_in_progress;

    for (; n < limit_epochs; ++n) {
        const Float_ epoch = n;
        const Float_ alpha = initial_alpha * (1.0 - epoch / num_epochs);

        int base_iteration = 0;
        std::fill(last_touched.begin(), last_touched.end(), -1);

        size_t i = 0;
        while (i < num_obs) {
            bool is_clear = true;
//            if (PRINT) { std::cout << "size is " << jobs_in_progress.size() << std::endl; }

            for (int t = jobs_in_progress.size(); t < nthreads && i < num_obs; ++t) {
                staging.get_alpha() = alpha;
                staging.get_observation() = i;

                // Tapping the RNG here in the serial section.
                auto& selections = staging.get_selections();
                selections.clear();
                auto& skips = staging.get_skips();
                skips.clear();

                const int self_iteration = i;
                constexpr unsigned char READONLY = 0;
                constexpr unsigned char WRITE = 1;

                {
                    auto& touched = last_touched[i];
                    auto& ttype = touch_type[i];
//                    if (PRINT) { std::cout << "SELF: " << i << ": " << touched << " (" << ttype << ")" << std::endl; }
                    if (touched >= base_iteration) {
                        is_clear = false;
//                        if (PRINT) { std::cout << "=== FAILED! ===" << std::endl; }
                    }
                    touched = self_iteration;
                    ttype = WRITE;
                }

                const size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
                for (size_t j = start; j < end; ++j) {
                    bool skip = setup.epoch_of_next_sample[j] > epoch;
                    skips.push_back(skip);
                    if (skip) {
                        continue;
                    }

                    {
                        auto neighbor = setup.tail[j];
                        auto& touched = last_touched[neighbor];
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

                    const size_t num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) * 
                        setup.negative_sample_rate / setup.epochs_per_sample[j]; 

                    for (size_t p = 0; p < num_neg_samples; ++p) {
                        size_t sampled = aarand::discrete_uniform(rng, num_obs);
                        if (sampled == i) {
                            continue;
                        }
                        selections.push_back(sampled);

                        auto& touched = last_touched[sampled];
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

                    selections.push_back(-1);

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
                        if (s != static_cast<size_t>(-1)) {
                            auto& touched = last_touched[s];
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
                    jobs_in_progress.push_back(thread_index);
                } else {
                    staging.run_direct();
                    staging.transfer_coordinates();
                }

                ++i;
            }

            // Waiting for all the jobs that were submitted.
            for (auto job : jobs_in_progress) {
                pool[job].wait();
                pool[job].transfer_coordinates();
            }
            jobs_in_progress.clear();

//            if (PRINT) { std::cout << "###################### OK ##########################" << std::endl; }

            base_iteration = i;
            if (!is_clear) {
                const int thread_index = i % nthreadsm1;
                pool[thread_index].migrate_parameters(staging);
                pool[thread_index].run();
                jobs_in_progress.push_back(thread_index);
                ++i;
            }
        }

        for (auto job : jobs_in_progress) {
            pool[job].wait();
            pool[job].transfer_coordinates();
        }
        jobs_in_progress.clear();
    }

    return;
#else
    throw std::runtime_error("umappp was not compiled with support for parallel optimization");
#endif
}

}

}

#endif
