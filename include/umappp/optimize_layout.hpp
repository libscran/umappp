#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#ifndef UMAPPP_NO_PARALLEL_OPTIMIZATION
#include <thread>
#include <atomic>
#endif

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

/*****************************************************
 ***************** Serial code ***********************
 *****************************************************/

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

                {
                    Float* right = embedding + setup.tail[j] * ndim;
                    Float dist2 = quick_squared_distance(left, right, ndim);
                    const Float pd2b = std::pow(dist2, b);
                    const Float grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));

                    Float* lcopy = left;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++right) {
                        Float gradient = alpha * clamp(grad_coef * (*lcopy - *right));
                        *lcopy += gradient;
                        *right -= gradient;
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

                    const Float* right = embedding + sampled * ndim;
                    Float dist2 = quick_squared_distance(left, right, ndim);
                    const Float grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));

                    Float* lcopy = left;
                    for (int d = 0; d < ndim; ++d, ++lcopy, ++right) {
                        *lcopy += alpha * clamp(grad_coef * (*lcopy - *right));
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
template<class Float, class Setup>
struct BusyWaiterThread {
public:
    std::vector<size_t> selections;
    std::vector<unsigned char> skips;
    size_t observation;
    Float alpha;

private:
    int ndim;
    Float* embedding;
    const Setup* setup;
    Float a;
    Float b;
    Float gamma;

    std::vector<Float> self_modified;

private:
    std::thread pool;
    std::atomic<bool> ready = false;
    bool finished = false;
    bool active = false;

public:
    void run() {
        ready.store(true, std::memory_order_release);
    }

    void wait() {
        while (ready.load(std::memory_order_acquire)) {
            ;
        }
    }

    void migrate_parameters(BusyWaiterThread& src) {
        selections.swap(src.selections);
        skips.swap(src.skips);
        alpha = src.alpha;
        observation = src.observation;
    }

    void transfer_coordinates() {
        std::copy(self_modified.begin(), self_modified.end(), embedding + observation * ndim);
    }

public:
    void run_direct() {
        auto seIt = selections.begin();
        auto skIt = skips.begin();
        const size_t i = observation;
        const size_t start = (i == 0 ? 0 : setup->head[i-1]), end = setup->head[i];

        // Copying it over into a thread-local buffer to avoid false sharing.
        // We don't bother doing this for the neighbors, though, as it's 
        // tedious to make sure that the modified values are available during negative sampling.
        // (This isn't a problem for the self, as the self cannot be its own negative sample.)
        {
            const Float* left = embedding + i * ndim;
            std::copy(left, left + ndim, self_modified.data());
        }

        for (size_t j = start; j < end; ++j) {
            if (*(skIt++)) {
                continue;
            }

            {
                Float* left = self_modified.data();
                Float* right = embedding + setup->tail[j] * ndim;

                Float dist2 = quick_squared_distance(left, right, ndim);
                const Float pd2b = std::pow(dist2, b);
                const Float grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));

                for (int d = 0; d < ndim; ++d, ++left, ++right) {
                    Float gradient = alpha * clamp(grad_coef * (*left - *right));
                    *left += gradient;
                    *right -= gradient;
                }
            }

            while (seIt != selections.end() && *seIt != -1) {
                Float* left = self_modified.data();
                const Float* right = embedding + (*seIt) * ndim;

                Float dist2 = quick_squared_distance(left, right, ndim);
                const Float grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));

                for (int d = 0; d < ndim; ++d, ++left, ++right) {
                    *left += alpha * clamp(grad_coef * (*left - *right));
                }
                ++seIt;
            }
            ++seIt; // get past the -1.
        }
    }

private:
    void loop() {
        while (true) {
            while (!ready.load(std::memory_order_acquire)) {
                ;
            }
            if (finished) {
                break;
            }
            run_direct();
            ready.store(false, std::memory_order_release);
        }
    }

public:
    BusyWaiterThread() {}

    BusyWaiterThread(int ndim_, Float* embedding_, Setup& setup_, Float a_, Float b_, Float gamma_) : 
        ndim(ndim_),
        embedding(embedding_),
        setup(&setup_),
        a(a_), 
        b(b_),
        gamma(gamma_),
        self_modified(ndim)
    {}

    void start() {
        active = true;
        pool = std::thread(&BusyWaiterThread::loop, this);
    }

public:
    ~BusyWaiterThread() {
        if (active) {
            finished = true;
            ready.store(true, std::memory_order_release);
            pool.join();
        }
    }

    BusyWaiterThread(BusyWaiterThread&&) = default;
    BusyWaiterThread& operator=(BusyWaiterThread&&) = default;

    BusyWaiterThread(const BusyWaiterThread& src) :
        selections(src.selections),
        skips(src.skips),
        observation(src.observation),

        ndim(src.ndim),
        embedding(src.embedding),
        setup(src.setup),
        a(src.a), 
        b(src.b),
        gamma(src.gamma),
        alpha(src.alpha),

        self_modified(src.self_modified)
    {}

    BusyWaiterThread& operator=(const BusyWaiterThread& src) {
        selections = src.selections;
        skips = src.skips;
        observation = src.observation;

        ndim = src.ndim;
        embedding = src.embedding;
        setup = src.setup;
        a = src.a; 
        b = src.b;
        gamma = src.gamma;
        alpha = src.alpha;

        self_modified = src.self_modified;
    }
};
#endif

//#define PRINT false

template<typename Float, class Setup, class Rng>
void optimize_layout_parallel(
    int ndim,
    Float* embedding, 
    Setup& setup,
    Float a, 
    Float b, 
    Float gamma,
    Float initial_alpha,
    Rng& rng,
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
    BusyWaiterThread<Float, Setup> staging(ndim, embedding, setup, a, b, gamma);

    int nthreadsm1 = nthreads - 1;
    std::vector<BusyWaiterThread<Float, Setup> > pool;
    pool.reserve(nthreadsm1);
    for (int t = 0; t < nthreadsm1; ++t) {
        pool.emplace_back(ndim, embedding, setup, a, b, gamma);
        pool.back().start();
    }

    std::vector<int> jobs_in_progress;

    for (; n < limit_epochs; ++n) {
        const Float epoch = n;
        const Float alpha = initial_alpha * (1.0 - epoch / num_epochs);

        int base_iteration = 0;
        std::fill(last_touched.begin(), last_touched.end(), -1);

        size_t i = 0;
        while (i < num_obs) {
            bool is_clear = true;
//            if (PRINT) { std::cout << "size is " << jobs_in_progress.size() << std::endl; }

            for (int t = jobs_in_progress.size(); t < nthreads && i < num_obs; ++t) {
                staging.alpha = alpha;
                staging.observation = i;

                // Tapping the RNG here in the serial section.
                auto& selections = staging.selections;
                selections.clear();
                auto& skips = staging.skips;
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
                        if (s != -1) {
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

#endif
