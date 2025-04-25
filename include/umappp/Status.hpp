#ifndef UMAPPP_STATUS_HPP
#define UMAPPP_STATUS_HPP

#include "Options.hpp"
#include "optimize_layout.hpp"

#include <random>
#include <cstddef>

/**
 * @file Status.hpp
 * @brief Status of the UMAP algorithm.
 */

namespace umappp {

/**
 * @brief Status of the UMAP optimization iterations.
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Float_ Floating-point type for the distances.
 *
 * Instances of this class should not be constructed directly, but instead returned by `initialize()`.
 */
template<typename Index_, typename Float_>
class Status {
public:
    /**
     * @cond
     */
    Status(internal::EpochData<Index_, Float_> epochs, Options options, std::size_t num_dim, Float_* embedding) : 
        my_epochs(std::move(epochs)),
        my_options(std::move(options)),
        my_engine(my_options.seed),
        my_num_dim(num_dim),
        my_embedding(embedding) 
    {}
    /**
     * @endcond
     */

private:
    internal::EpochData<Index_, Float_> my_epochs;
    Options my_options;
    std::mt19937_64 my_engine;
    std::size_t my_num_dim;
    Float_* my_embedding;

public:
    /**
     * @cond
     */
    // Only used for testing, specifically in comparison to uwot:::data2set.
    const auto& get_epoch_data() const {
        return my_epochs;
    }
    /**
     * @endcond
     */

public:
    /**
     * @return Number of dimensions of the embedding.
     */
    std::size_t num_dimensions() const {
        return my_num_dim;
    }

    /**
     * @return Pointer to an array containing the embeddings after the specified number of epochs.
     * This is a column-major matrix where rows are dimensions (`num_dimensions()`) and columns are observations (`num_observations()`).
     */
    const Float_* embedding() const {
        return my_embedding;
    }

    /**
     * @param ptr Pointer to an array as described in `embedding()`.
     * @param copy Whether the contents of the previous array should be copied into `ptr`.
     *
     * By default, the `Status` object will operate on embeddings in an array specified at its own construction time.
     * This method will change the embedding array for an existing `Status` object, which can be helpful in some situations, 
     * e.g., to clone a `Status` object and to store its embeddings in a different array than the object.
     *
     * The contents of the new array in `ptr` should be the same as the array that it replaces, as `run()` will continue the iteration from the coordinates inside the array.
     * This is enforced by default when `copy = true`, but if the supplied `ptr` already contains a copy, the caller may set `copy = false` to avoid extra work
     */
    void set_embedding(Float_* ptr, bool copy = true) {
        if (copy) {
            std::size_t n = num_dimensions() * static_cast<std::size_t>(num_observations()); // cast to avoid overflow.
            std::copy_n(my_embedding, n, ptr);
        }
        my_embedding = ptr;
    }

    /**
     * @return Current epoch.
     */
    int epoch() const {
        return my_epochs.current_epoch;
    }

    /**
     * @return Total number of epochs.
     * This is typically determined by the value of `Options::max_epochs` used in `initialize()`.
     */
    int num_epochs() const {
        return my_epochs.total_epochs;
    }

    /**
     * @return The number of observations in the dataset.
     */
    Index_ num_observations() const {
        return my_epochs.head.size();
    }

public:
    /** 
     * The status of the algorithm and the coordinates in `embedding()` are updated to the specified number of epochs. 
     *
     * @param epoch_limit Number of epochs to run to.
     * The actual number of epochs performed is equal to the difference between `epoch_limit` and the current number of epochs in `epoch()`.
     * `epoch_limit` should be not less than `epoch()` and be no greater than the maximum number of epochs specified in `max_epochs()`.
     * If zero, defaults to the maximum number of epochs. 
     */
    void run(int epoch_limit = 0) {
        if (epoch_limit == 0) {
            epoch_limit = my_epochs.total_epochs;
        }

        if (my_options.num_threads == 1 || !my_options.parallel_optimization) {
            internal::optimize_layout<Index_, Float_>(
                my_num_dim,
                my_embedding,
                my_epochs,
                my_options.a,
                my_options.b,
                my_options.repulsion_strength,
                my_options.learning_rate,
                my_engine,
                epoch_limit
            );
        } else {
            internal::optimize_layout_parallel<Index_, Float_>(
                my_num_dim,
                my_embedding,
                my_epochs,
                my_options.a,
                my_options.b,
                my_options.repulsion_strength,
                my_options.learning_rate,
                my_engine,
                epoch_limit,
                my_options.num_threads
            );
        }
        return;
    }
};

}

#endif
