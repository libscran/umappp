#ifndef UMAPPP_STATUS_HPP
#define UMAPPP_STATUS_HPP

#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "Options.hpp"
#include "optimize_layout.hpp"

/**
 * @file Status.hpp
 * @brief Status of the UMAP algorithm.
 */

namespace umappp {

/**
 * @brief Status of the UMAP optimization iterations.
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Float_ Floating-point type of the distances.
 *
 * Instances of this class should not be constructed directly, but instead returned by `initialize()`.
 */
template<typename Index_, typename Float_>
class Status {
public:
    /**
     * @cond
     */
    Status(internal::EpochData<Index_, Float_> epochs, Options options, const std::size_t num_dim) :
        my_epochs(std::move(epochs)),
        my_options(std::move(options)),
        my_engine(my_options.optimize_seed),
        my_num_dim(num_dim)
    {}
    /**
     * @endcond
     */

private:
    internal::EpochData<Index_, Float_> my_epochs;
    Options my_options;
    RngEngine my_engine;
    std::size_t my_num_dim;

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
     * @return Current epoch, i.e., the number of epochs that have already been performed by `run()`.
     */
    int epoch() const {
        return my_epochs.current_epoch;
    }

    /**
     * @return Total number of epochs that can be performed by `run()`.
     * This is typically determined by the value of `Options::num_epochs` used in `initialize()`.
     */
    int num_epochs() const {
        return my_epochs.total_epochs;
    }

    /**
     * @return The number of observations in the dataset.
     */
    Index_ num_observations() const {
        return my_epochs.cumulative_num_edges.size() - 1;
    }

public:
    /** 
     * The status of the algorithm and the coordinates in `embedding()` are updated to the specified number of epochs. 
     *
     * @param[in, out] embedding Pointer to an array containing a column-major matrix where rows are dimensions and columns are observations.
     * On input, this should contain the embeddings at the current epoch (`epoch()`),
     * and on output, this should contain the embedding at `epoch_limit`.
     * Typically, this should be the same array that was used in `initialize()`.
     * @param epoch_limit Number of epochs to run to.
     * The actual number of epochs performed is equal to the difference between `epoch_limit` and `epoch()`.
     * `epoch_limit` should be not less than `epoch()` and be no greater than the maximum number of epochs specified in `num_epochs()`.
     */
    void run(Float_* const embedding, int epoch_limit) {
        if (my_options.num_threads == 1 || !my_options.parallel_optimization) {
            internal::optimize_layout<Index_, Float_>(
                my_num_dim,
                embedding,
                my_epochs,
                *(my_options.a),
                *(my_options.b),
                my_options.repulsion_strength,
                my_options.learning_rate,
                my_engine,
                epoch_limit
            );
        } else {
            internal::optimize_layout_parallel<Index_, Float_>(
                my_num_dim,
                embedding,
                my_epochs,
                *(my_options.a),
                *(my_options.b),
                my_options.repulsion_strength,
                my_options.learning_rate,
                my_engine,
                epoch_limit,
                my_options.num_threads
            );
        }
    }

    /** 
     * The status of the algorithm and the coordinates in `embedding()` are updated after completing `num_epochs()`.
     *
     * @param[in, out] embedding Pointer to an array containing a column-major matrix where rows are dimensions and columns are observations.
     * On input, this should contain the embeddings at the current epoch (`epoch()`),
     * and on output, this should contain the embedding at `num_epochs()`.
     * Typically, this should be the same array that was used in `initialize()`.
     */
    void run(Float_* const embedding) {
        run(embedding, my_epochs.total_epochs);
    }
};

}

#endif
