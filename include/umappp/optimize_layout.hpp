#ifndef UMAPPP_OPTIMIZE_LAYOUT_HPP
#define UMAPPP_OPTIMIZE_LAYOUT_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <thread>

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
                    size_t sampled = p % num_obs;
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

template<class Float, class Setup>
struct PersistentThread {
public:
    std::vector<size_t> selections;
    std::vector<unsigned char> skips;
    size_t from, to;

    int ndim;
    Float* embedding;
    Setup* setup;
    Float a;
    Float b;
    Float gamma;
    Float alpha;

    bool active = false;
    std::mutex mut;
    std::condition_variable cv;
    bool finished = false;
    bool done = false;
    std::thread pool;

public:
    void loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(mut);
            cv.wait(lock, [this] { return finished; });
            if (finished) {
                break;
            }

            done = false;
            auto seIt = selections.begin();
            auto skIt = skips.begin();
            for (size_t i = from; i < to; ++i) {
                size_t start = (i == 0 ? 0 : setup->head[i-1]), end = setup->head[i];
                Float* left = embedding + i * ndim;

                for (size_t j = start; j < end; ++j) {
                    if (*(skIt++)) {
                        continue;
                    }

                    {
                        Float* right = embedding + setup->tail[j] * ndim;
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

                    while (seIt != selections.end() && *seIt != -1) {
                        const Float* right = embedding + (*seIt) * ndim;
                        Float dist2 = quick_squared_distance(left, right, ndim);
                        const Float grad_coef = 2 * gamma * b / ((0.001 + dist2) * (a * std::pow(dist2, b) + 1.0));
                        Float* lcopy = left;
                        for (int d = 0; d < ndim; ++d, ++lcopy, ++right) {
                            *lcopy += alpha * clamp(grad_coef * (*lcopy - *right));
                        }
                        ++seIt;
                    }
                    ++seIt; // get past the -1.
                }
            }

            done = true;
        }
    }

public:
    PersistentThread() {}

    PersistentThread(int ndim_, Float* embedding_, Setup& setup_, Float a_, Float b_, Float gamma_) : 
        ndim(ndim_),
        embedding(embedding_),
        setup(&setup_),
        a(a_), 
        b(b_),
        gamma(gamma_)
    {}

    void start() {
        active = true;
        pool = std::thread(&PersistentThread::loop, this);
    }

public:
    ~PersistentThread() {
        if (active) {
            {
                std::unique_lock<std::mutex> lock(mut);
                finished = true;
            }
            cv.notify_one();
            pool.join();
        }
    }

    PersistentThread(PersistentThread&&) = default;
    PersistentThread& operator=(PersistentThread&&) = default;

    PersistentThread(const PersistentThread& src) :
        selections(src.selections),
        skips(src.skips),
        from(src.from),
        to(src.to),

        ndim(src.ndim),
        embedding(src.embedding),
        setup(src.setup),
        a(src.a), 
        b(src.b),
        gamma(src.gamma),
        alpha(src.alpha)
    {}

    PersistentThread& operator=(const PersistentThread& src) {
        selections = src.selections;
        skips = src.skips;
        from = src.from;
        to = src.to;

        ndim = src.ndim;
        embedding = src.embedding;
        setup = src.setup;
        a = src.a; 
        b = src.b;
        gamma = src.gamma;
        alpha = src.alpha;
    }
};

template<typename Float, class Setup, class Rng>
void optimize_layout_concurrent(
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
    auto& n = setup.current_epoch;
    auto num_epochs = setup.total_epochs;
    auto limit_epochs = num_epochs;
    if (epoch_limit> 0) {
        limit_epochs = std::min(epoch_limit, num_epochs);
    }

    const size_t num_obs = setup.head.size(); 
    constexpr size_t block_size = 5;
    std::vector<int> last_touched(num_obs);
    std::vector<unsigned char> touch_type(num_obs);
    std::vector<int> jobs;

    std::vector<PersistentThread<Float, Setup> > pool;
    pool.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        pool.emplace_back(ndim, embedding, setup, a, b, gamma);
        pool.back().start();
    }

    for (; n < limit_epochs; ++n) {
        const Float epoch = n;
        const Float alpha = initial_alpha * (1.0 - epoch / num_epochs);

        size_t block_start = 0;
        int iteration = 0;
        std::fill(last_touched.begin(), last_touched.end(), -1);

        while (block_start < num_obs) {
            bool is_clear = true;
            int thread_count = jobs.size();

            while (thread_count < nthreads) {
                // Setting parameters for the current block.
                {
                    std::lock_guard<std::mutex> lock(pool[thread_count].mut);
        
                    auto& selections = pool[thread_count].selections;
                    selections.clear();
                    auto& skips = pool[thread_count].skips;
                    skips.clear();
                    int self_iteration = iteration + thread_count;

                    size_t block_end = std::min(block_start + block_size, num_obs);
                    pool[thread_count].alpha = alpha;
                    pool[thread_count].from = block_start;
                    pool[thread_count].to = block_end;

                    for (size_t i = block_start; i < block_end; ++i) {
                        size_t start = (i == 0 ? 0 : setup.head[i-1]), end = setup.head[i];
                        {
                            auto& touched = last_touched[i];
                            auto& ttype = touch_type[i];
    //                        std::cout << "\tSELF\t" << i << "\t" << touched << "\t" << iteration << "\t" << self_iteration << std::endl;
                            if (touched >= iteration) {
                                if (touched != self_iteration) {
    //                                std::cout << "FAILED!" << std::endl;
                                    is_clear = false;
                                }
                            } else {
                                touched = self_iteration;
                                ttype = 1;
                            }
                        }

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
    //                            std::cout << "\tFRIEND\t" << neighbor << "\t" << touched << "\t" << iteration << "\t" << self_iteration << std::endl;
                                if (touched >= iteration) {
                                    if (touched != self_iteration) {
    //                                    std::cout << "FAILED!" << std::endl;
                                        is_clear = false;
                                    }
                                } else {
                                    touched = self_iteration;
                                    ttype = 1;
                                }
                            }

                            const size_t num_neg_samples = (epoch - setup.epoch_of_next_negative_sample[j]) * 
                                setup.negative_sample_rate / setup.epochs_per_sample[j]; // i.e., 1/epochs_per_negative_sample.

                            for (size_t p = 0; p < num_neg_samples; ++p) {
                                size_t sampled = aarand::discrete_uniform(rng, num_obs);
                                if (sampled == i) {
                                    continue;
                                }
                                selections.push_back(sampled);

                                auto& touched = last_touched[sampled];
                                auto& ttype = touch_type[sampled];
    //                            std::cout << "\tSAMPLED\t" << sampled << "\t" << touched << "\t" << iteration << "\t" << self_iteration << std::endl;
                                if (touched > iteration) {
                                    if (touched != self_iteration) {
                                        if (ttype == 1) {
    //                                        std::cout << "FAILED!" << std::endl;
                                            is_clear = false;
                                        }
                                    }
                                } else {
                                    touched = self_iteration;
                                    ttype = 0;
                                }
                            }
                            selections.push_back(-1);

                            setup.epoch_of_next_sample[j] += setup.epochs_per_sample[j];
                            setup.epoch_of_next_negative_sample[j] = epoch;
                        }
                    }
                }

                if (!is_clear) {
                    break;
                } 

                // Submitting the job.
                pool[thread_count].cv.notify_one();
                ++thread_count;
                block_start += block_size;
            }

            // Waiting for all the jobs that were submitted.
            for (auto job : jobs) {
                std::unique_lock lock(pool[job].mut);
                pool[job].cv.wait(lock, [&] { return pool[job].done; });
            }
            jobs.clear();

//            std::cout << thread_count << "\t" << block_start << " - " << block_end << std::endl;
            if (is_clear) {
                iteration += nthreads;
            } else {
                pool[thread_count].cv.notify_one();
                iteration += thread_count; 
                block_start += block_size;
            }
        }

        for (auto job : jobs) {
            std::unique_lock lock(pool[job].mut);
            pool[job].cv.wait(lock, [&] { return pool[job].done; });
        }
    }

    return;
}

}

#endif
