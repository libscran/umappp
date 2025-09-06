#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#ifndef UMAPPP_NO_PARALLEL_OPTIMIZATION
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#else
#include <stdexcept>
#endif

#include "aarand/aarand.hpp"
#include "sanisizer/sanisizer.hpp"

#include "NeighborList.hpp"

namespace umappp {

namespace internal {

template<typename Input_>
std::remove_cv_t<std::remove_reference_t<Input_> > I(const Input_ x) {
    return x;
}

template<typename Index_, typename Float_>
struct EpochData {
    EpochData(const Index_ nobs) : cumulative_num_edges(sanisizer::sum<decltype(I(cumulative_num_edges.size()))>(nobs, 1)) {}

    int total_epochs;
    int current_epoch = 0;

    // Store the graph as a (symmetric) compressed sparse matrix.
    // Cumulative_num_edges is the equivalent to indptrs while edge_targets are the indices.
    // The various 'epochs_per_*' and 'epoch_of_*' vectors are the values/edge weights.
    std::vector<std::size_t> cumulative_num_edges;
    std::vector<Index_> edge_targets;

    std::vector<Float_> epochs_per_sample;
    std::vector<Float_> epoch_of_next_sample;

    std::vector<Float_> epoch_of_next_negative_sample;
    Float_ negative_sample_rate;
};

template<typename Index_, typename Float_>
EpochData<Index_, Float_> similarities_to_epochs(const NeighborList<Index_, Float_>& p, const int num_epochs, const Float_ negative_sample_rate) {
    Float_ maxed = 0;
    std::size_t count = 0;
    for (const auto& x : p) {
        count = sanisizer::sum<std::size_t>(count, x.size());
        for (const auto& y : x) {
            maxed = std::max(maxed, y.second);
        }
    }

    const Index_ num_obs = p.size(); // Index_ should be able to hold the number of observations.
    EpochData<Index_, Float_> output(num_obs);
    output.total_epochs = num_epochs;
    output.edge_targets.reserve(count);
    output.epochs_per_sample.reserve(count);
    const Float_ limit = maxed / num_epochs;

    for (Index_ i = 0; i < num_obs; ++i) {
        const auto& x = p[i];
        for (const auto& y : x) {
            if (y.second >= limit) {
                output.edge_targets.push_back(y.first);
                output.epochs_per_sample.push_back(maxed / y.second);
            }
        }
        output.cumulative_num_edges[i + 1] = output.edge_targets.size();
    }

    // Filling in some epoch-related running statistics.
    output.epoch_of_next_sample = output.epochs_per_sample;
    output.epoch_of_next_negative_sample = output.epochs_per_sample;
    for (auto& e : output.epoch_of_next_negative_sample) {
        e /= negative_sample_rate;
    }
    output.negative_sample_rate = negative_sample_rate;

    // Maximum value of 'num_neg_samples' should be 'num_epochs * negative_sample_rate', because:
    // - '(epoch - setup.epoch_of_next_negative_sample[j])' has a maximum value of 'num_epochs'.
    // - 'epochs_per_negative_sample' has a minimum value of '1/negative_sample_rate', because:
    // - 'setup.epochs_per_sample[j]' has a minimum value of 1, when 'y.second == maxed'.
    // So we just have to check that the cast is safe once, for the maximum value.
    sanisizer::from_float<int>(static_cast<Float_>(num_epochs) * negative_sample_rate);

    return output;       
}

template<typename Float_>
Float_ quick_squared_distance(const Float_* const left, const Float_* const right, const std::size_t num_dim) {
    Float_ dist2 = 0;
    for (std::size_t d = 0; d < num_dim; ++d) {
        Float_ delta = (left[d] - right[d]);
        dist2 += delta * delta;
    }
    constexpr Float_ dist_eps = std::numeric_limits<Float_>::epsilon();
    return std::max(dist_eps, dist2);
}

template<typename Float_>
Float_ clamp(const Float_ input) {
    constexpr Float_ min_gradient = -4;
    constexpr Float_ max_gradient = 4;
    return std::min(std::max(input, min_gradient), max_gradient);
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
    const auto num_epochs = setup.total_epochs;

    for (; n < epoch_limit; ++n) {
        const Float_ epoch = n;
        const Float_ alpha = initial_alpha * (1.0 - epoch / num_epochs);

        const Index_ num_obs = setup.cumulative_num_edges.size() - 1; 
        for (Index_ i = 0; i < num_obs; ++i) {
            const auto start = setup.cumulative_num_edges[i], end = setup.cumulative_num_edges[i + 1];
            const auto left = embedding + sanisizer::product_unsafe<std::size_t>(i, num_dim);

            for (auto j = start; j < end; ++j) {
                if (setup.epoch_of_next_sample[j] > epoch) {
                    continue;
                }

                {
                    const auto right = embedding + sanisizer::product_unsafe<std::size_t>(setup.edge_targets[j], num_dim);
                    const Float_ dist2 = quick_squared_distance(left, right, num_dim);
                    const Float_ pd2b = std::pow(dist2, b);
                    const Float_ grad_coef = (-2 * a * b * pd2b) / (dist2 * (a * pd2b + 1.0));

                    for (std::size_t d = 0; d < num_dim; ++d) {
                        auto& l = left[d];
                        auto& r = right[d];
                        const Float_ gradient = alpha * clamp(grad_coef * (l - r));
                        l += gradient;
                        r -= gradient;
                    }
                }

                const Float_ epochs_per_negative_sample = setup.epochs_per_sample[j] / setup.negative_sample_rate;
                const int num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) / epochs_per_negative_sample; // cast is known to be safe, see initialize().

                for (int p = 0; p < num_neg_samples; ++p) {
                    const auto sampled = aarand::discrete_uniform(rng, num_obs);
                    if (sampled == i) {
                        continue;
                    }

                    const auto right = embedding + sanisizer::product_unsafe<std::size_t>(sampled, num_dim);
                    const Float_ dist2 = quick_squared_distance(left, right, num_dim);
                    const Float_ grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));

                    for (std::size_t d = 0; d < num_dim; ++d) {
                        left[d] += alpha * clamp(grad_coef * (left[d] - right[d]));
                    }
                }

                setup.epoch_of_next_sample[j] += setup.epochs_per_sample[j];
                setup.epoch_of_next_negative_sample[j] += num_neg_samples * epochs_per_negative_sample;
            }
        }
    }

    return;
}

/*****************************************************
 **************** Parallel code **********************
 *****************************************************/

#ifndef UMAPPP_NO_PARALLEL_OPTIMIZATION
constexpr int skip_ns_sentinel = -1;

template<typename Index_, typename Float_>
struct BusyWaiterInput {
    std::vector<Index_> negative_sample_selections;
    std::vector<int> negative_sample_count;
    Index_ observation;
    std::size_t edge_target_index_start;
    Float_ alpha;
};

template<typename Index_, typename Float_>
struct BusyWaiterState {
    std::size_t num_dim;
    Float_* embedding;
    const EpochData<Index_, Float_>* setup;
    Float_ a;
    Float_ b;
    Float_ gamma;
    std::vector<Float_> self_modified;
};

template<typename Index_, typename Float_>
void optimize_single_observation(const BusyWaiterInput<Index_, Float_>& input, BusyWaiterState<Index_, Float_>& state) {
    // Copying it over into a thread-local buffer to avoid false sharing.
    // We don't bother doing this for the neighbors, though, as it's 
    // tedious to make sure that the modified values are available during negative sampling.
    // (This isn't a problem for the self, as the self cannot be its own negative sample.)
    const auto source = state.embedding + sanisizer::product_unsafe<std::size_t>(input.observation, state.num_dim);
    std::copy_n(source, state.num_dim, state.self_modified.data());

    decltype(I(input.negative_sample_selections.size())) position = 0;
    const auto num_neighbors = input.negative_sample_count.size();
    for (decltype(I(num_neighbors)) n = 0; n < num_neighbors; ++n) {
        const auto number = input.negative_sample_count[n];
        if (number == skip_ns_sentinel) {
            continue;
        }

        {
            const auto left = state.self_modified.data();
            const auto j = sanisizer::sum_unsafe<std::size_t>(n, input.edge_target_index_start);
            const auto right = state.embedding + sanisizer::product_unsafe<std::size_t>(state.setup->edge_targets[j], state.num_dim);

            const Float_ dist2 = quick_squared_distance(left, right, state.num_dim);
            const Float_ pd2b = std::pow(dist2, state.b);
            const Float_ grad_coef = (-2 * state.a * state.b * pd2b) / (dist2 * (state.a * pd2b + 1.0));

            for (std::size_t d = 0; d < state.num_dim; ++d) {
                auto& l = left[d];
                auto& r = right[d];
                const Float_ gradient = input.alpha * clamp(grad_coef * (l - r));
                l += gradient;
                r -= gradient;
            }
        }

        auto s = position;
        position += number;
        for (; s < position; ++s) {
            const auto left = state.self_modified.data();
            const auto right = state.embedding + sanisizer::product_unsafe<std::size_t>(input.negative_sample_selections[s], state.num_dim);

            const Float_ dist2 = quick_squared_distance(left, right, state.num_dim);
            const Float_ grad_coef = 2 * state.gamma * state.b / ((0.001 + dist2) * (state.a * std::pow(dist2, state.b) + 1.0));

            for (std::size_t d = 0; d < state.num_dim; ++d) {
                left[d] += input.alpha * clamp(grad_coef * (left[d] - right[d]));
            }
        }
    }

    // Copying it back to the embedding once we're done.
    std::copy(state.self_modified.begin(), state.self_modified.end(), source);
}

template<typename Index_, typename Float_>
class BusyWaiterThread {
private:
    struct SyncData {
        std::atomic<bool> ready = false;
        bool finished = false;
        bool active = false;
    };

    SyncData* my_sync;
    std::thread my_worker;
    BusyWaiterInput<Index_, Float_>* my_input;

public:
    void run(BusyWaiterInput<Index_, Float_>& input) {
        my_input = &input;
        my_sync->ready.store(true, std::memory_order_release);
    }

    void wait() {
        while (my_sync->ready.load(std::memory_order_acquire)) {
            ;
        }
    }

public:
    BusyWaiterThread(const BusyWaiterState<Index_, Float_>& x) {
        std::mutex init_mut;
        std::condition_variable init_cv;
        bool initialized = false;

        my_worker = std::thread([&]() -> void {
            SyncData sync; // Allocating within each thread to reduce false sharing.
            BusyWaiterState<Index_, Float_> state(x); // Make a copy to reduce false sharing.

            {
                std::lock_guard ilck(init_mut);
                initialized = true;
                my_sync = &sync;
                init_cv.notify_one();
            }

            while (true) {
                while (!sync.ready.load(std::memory_order_acquire)) {
                    ;
                }
                if (sync.finished) {
                    break;
                }
                optimize_single_observation(*my_input, state); // this had better be noexcept... no memory allocations, just math.
                sync.ready.store(false, std::memory_order_release);
            }
        });

        std::unique_lock ilck(init_mut);
        init_cv.wait(ilck, [&]() -> bool { return initialized; });
    }
 
public:
    ~BusyWaiterThread() {
        if (my_sync != NULL) {
            wait();
            my_sync->finished = true;
            my_sync->ready.store(true, std::memory_order_release);
        }
        my_worker.join();
    }

    BusyWaiterThread& operator=(BusyWaiterThread&&) = default;
    BusyWaiterThread(BusyWaiterThread&&) = default;
    BusyWaiterThread& operator=(const BusyWaiterThread&) = delete;
    BusyWaiterThread(const BusyWaiterThread&) = delete;
};
#endif

//#define PRINT false

template<typename Index_, typename Float_, class Rng_>
void optimize_layout_parallel(
    const std::size_t num_dim,
    Float_* const embedding, 
    EpochData<Index_, Float_>& setup,
    const Float_ a, 
    const Float_ b, 
    const Float_ gamma,
    const Float_ initial_alpha,
    Rng_& rng,
    const int epoch_limit,
    const int nthreads
) {
#ifndef UMAPPP_NO_PARALLEL_OPTIMIZATION
    auto& n = setup.current_epoch;
    const auto num_epochs = setup.total_epochs;

    BusyWaiterState<Index_, Float_> state;
    state.num_dim = num_dim;
    state.embedding = embedding;
    state.setup = &setup;
    state.a = a;
    state.b = b;
    state.gamma = gamma;
    state.self_modified.resize(state.num_dim);

    // We use 'nthreads - 1' busy waiters so that some work runs on the main
    // thread. This ensures that we don't spin off 'nthreads' and then have the
    // main thread running the spin lock to compete for CPU usage. Instead, if
    // all threads are in use, the main thread is also doing useful work.
    std::vector<BusyWaiterThread<Index_, Float_> > pool;
    pool.reserve(nthreads - 1);
    for (int t = 0; t < nthreads - 1; ++t) {
        pool.emplace_back(state);
    }

    auto raw_inputs = sanisizer::create<std::vector<BusyWaiterInput<Index_, Float_> > >(nthreads);
    BusyWaiterInput<Index_, Float_>* main_input = &(raw_inputs.back());
    std::vector<BusyWaiterInput<Index_, Float_>*> pool_inputs;
    pool_inputs.reserve(nthreads - 1);
    for (int t = 0; t < nthreads - 1; ++t) {
        pool_inputs.push_back(&(raw_inputs[t]));
    }

    const Index_ num_obs = setup.cumulative_num_edges.size() - 1; 
    std::vector<Index_> last_touched_iteration(num_obs);
    std::vector<unsigned char> touch_type(num_obs);

    for (; n < epoch_limit; ++n) {
        const Float_ epoch = n;
        const Float_ alpha = initial_alpha * (1.0 - epoch / num_epochs);

        // Iteration is 1-based so as to allow last_touched_iteration[i] = 0 to
        // mean that it has never been touched. A touch after the first
        // iteration is instead last_touched_iteration[i] = 1. (We can't use
        // last_touched_iteration[i] = -1 because Index_ might be unsigned.)
        Index_ base_iteration = 1;
        std::fill(last_touched_iteration.begin(), last_touched_iteration.end(), 0);

        int used_threads = 0;
        Index_ i = 0;
        while (i < num_obs) {
            bool is_clear = true;
//            if (PRINT) { std::cout << "size is " << threads_in_progress.size() << std::endl; }

            for (int t = used_threads; t < nthreads; ++t) {
                auto& input = *main_input;
                input.alpha = alpha;
                input.observation = i;

                // Tapping the RNG here in the serial section.
                auto& ns_selections = input.negative_sample_selections;
                ns_selections.clear();
                auto& ns_count = input.negative_sample_count;
                ns_count.clear();

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

                const auto start = setup.cumulative_num_edges[i], end = setup.cumulative_num_edges[i + 1];
                input.edge_target_index_start = start;
                for (auto j = start; j < end; ++j) {
                    if (setup.epoch_of_next_sample[j] > epoch) {
                        ns_count.push_back(skip_ns_sentinel);
                        continue;
                    }

                    {
                        const auto neighbor = setup.edge_targets[j];
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

                    const auto prior_size = ns_selections.size();
                    const Float_ epochs_per_negative_sample = setup.epochs_per_sample[j] / setup.negative_sample_rate;
                    const int num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) / epochs_per_negative_sample; // cast is known to be safe, see initialize().

                    for (int p = 0; p < num_neg_samples; ++p) {
                        const Index_ sampled = aarand::discrete_uniform(rng, num_obs);
                        if (sampled == i) {
                            continue;
                        }
                        ns_selections.push_back(sampled);

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

                    ns_count.push_back(ns_selections.size() - prior_size);
                    setup.epoch_of_next_sample[j] += setup.epochs_per_sample[j];
                    setup.epoch_of_next_negative_sample[j] += num_neg_samples * epochs_per_negative_sample;
                }

                if (!is_clear) {
                    // As we only updated the access for 'sampled' to READONLY
                    // if they weren't touched by another thread, we need to go
                    // through and manually update them now that the next round
                    // of thread_iterations will use 'self_iteration' as the
                    // 'base_iteration'. This ensures that the flags are properly
                    // set for the next round, under the expectation that the
                    // pending thread becomes the first thread.
                    for (auto s : ns_selections) {
                        auto& touched = last_touched_iteration[s];
                        if (touched != self_iteration) {
                            touched = self_iteration;
                            touch_type[s] = READONLY;
                        }
                    }
                    break;
                } 

                if (t + 1 == nthreads) {
                    // If we saturate the number of threads, we run the last task
                    // on the main thread to ensure that the main thread's spinlock
                    // won't compete other threads for with CPU time.
                    optimize_single_observation(*main_input, state);
                } else {
                    std::swap(pool_inputs[t], main_input);
                    pool[t].run(*(pool_inputs[t]));
                    ++used_threads;
                }

                ++i;
                if (i == num_obs) {
                    break;
                }
            }

            // Waiting for all the jobs that were submitted.
            for (int t = 0; t < used_threads; ++t) {
                pool[t].wait();
            }

//            if (PRINT) { std::cout << "###################### OK ##########################" << std::endl; }

            base_iteration = i + 1; // remember, iterations are 1-based.

            // If an observation has a conflict that causes us to break out
            // early, we launch its job on the first thread once all the
            // previous conflicting jobs have finished.
            if (!is_clear) {
                std::swap(pool_inputs[0], main_input);
                pool[0].run(*(pool_inputs[0]));
                used_threads = 1;
                ++i;
            } else {
                used_threads = 0;
            }
        }

        for (int t = 0; t < used_threads; ++t) {
            pool[t].wait();
        }
    }

    return;
#else
    throw std::runtime_error("umappp was not compiled with support for parallel optimization");
#endif
}

}

}

#endif
