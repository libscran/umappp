#ifndef UMAPPP_INITIALIZE_HPP
#define UMAPPP_INITIALIZE_HPP

#include "NeighborList.hpp"
#include "combine_neighbor_sets.hpp"
#include "find_ab.hpp"
#include "neighbor_similarities.hpp"
#include "spectral_init.hpp"
#include "Status.hpp"

#include "knncolle/knncolle.hpp"

#include <random>
#include <cstddef>

/**
 * @file initialize.hpp
 * @brief Initialize the UMAP algorithm.
 */

namespace umappp {

/**
 * @cond
 */
namespace internal {

template<typename Index_>
int choose_num_epochs(const int num_epochs, const Index_ size) {
    if (num_epochs >= 0) {
        return num_epochs;
    }

    // Choosing the number of epochs. We use a simple formula to decrease
    // the number of epochs with increasing size, with the aim being that
    // the 'extra work' beyond the minimal 200 epochs should be the same
    // regardless of the number of observations. Given one calculation per
    // observation per epoch, this amounts to 300 * 10000 calculations at
    // the lower bound, so we simply choose a number of epochs that
    // equalizes the number of calculations for any number of observations.
    constexpr Index_ limit = 10000;
    const int minimal = 200, maximal = 300;
    if (size <= limit) {
        return minimal + maximal;
    } else {
        return minimal + static_cast<int>(std::ceil(maximal * static_cast<double>(limit) / static_cast<double>(size)));
    }
}

}
/**
 * @endcond
 */

/** 
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Float_ Floating-point type for the distances.
 *
 * @param x Indices and distances to the nearest neighbors for each observation.
 * Note the expectations in the `NeighborList` documentation.
 * @param num_dim Number of dimensions of the embedding.
 * @param[in, out] embedding Pointer to an array in which to store the embedding.
 * This is treated as a column-major matrix where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * Existing values in this array will be used as input if `Options::initialize = InitializeMethod::NONE`, 
 * and may be used as input if `Options::initialize = InitializeMethod::SPECTRAL` and `Options::initialize_random_on_spectral_fail = false`;
 * otherwise it is only used as output.
 * @param options Further options.
 * Note that `Options::num_neighbors` is ignored here.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 * Further calls to `Status::run()` will update the embeddings in `embedding`.
 */
template<typename Index_, typename Float_>
Status<Index_, Float_> initialize(NeighborList<Index_, Float_> x, const std::size_t num_dim, Float_* const embedding, Options options) {
    internal::NeighborSimilaritiesOptions<Float_> nsopt;
    nsopt.local_connectivity = options.local_connectivity;
    nsopt.bandwidth = options.bandwidth;
    nsopt.num_threads = options.num_threads;
    internal::neighbor_similarities(x, nsopt);

    internal::combine_neighbor_sets(x, static_cast<Float_>(options.mix_ratio));

    bool use_random = (options.initialize_method == InitializeMethod::RANDOM);
    if (options.initialize_method == InitializeMethod::SPECTRAL) {
        const bool spectral_okay = internal::spectral_init(
            x,
            num_dim,
            embedding,
            options.num_threads,
            options.initialize_spectral_scale,
            options.initialize_spectral_jitter,
            options.initialize_spectral_jitter_sd,
            options.initialize_seed
        );
        use_random = (options.initialize_random_on_spectral_fail && !spectral_okay);
    }

    if (use_random) {
        internal::random_init<Index_>(
            x.size(),
            num_dim,
            embedding,
            options.initialize_seed,
            options.initialize_random_scale
        );
    }

    // Finding a good a/b pair.
    if (options.a <= 0 || options.b <= 0) {
        const auto found = internal::find_ab(options.spread, options.min_dist);
        options.a = found.first;
        options.b = found.second;
    }

    options.num_epochs = internal::choose_num_epochs<Index_>(options.num_epochs, x.size());

    return Status<Index_, Float_>(
        internal::similarities_to_epochs<Index_, Float_>(x, options.num_epochs, options.negative_sample_rate),
        options,
        num_dim
    );
}

/**
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Input_ Floating-point type of the input data for the neighbor search.
 * This is not used other than to define the `knncolle::Prebuilt` type.
 * @tparam Float_ Floating-point type of the input data, neighbor distances and output embedding.
 *
 * @param prebuilt A neighbor search index built on the dataset of interest.
 * @param num_dim Number of dimensions of the UMAP embedding.
 * @param[in, out] embedding Pointer to an array in which to store the embedding.
 * This is treated as a column-major matrix where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * Existing values in this array will be used as input if `Options::initialize = InitializeMethod::NONE`, 
 * and may be used as input if `Options::initialize = InitializeMethod::SPECTRAL` and `Options::initialize_random_on_spectral_fail = false`;
 * otherwise it is only used as output.
 * @param options Further options.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 * Further calls to `Status::run()` will update the embeddings in `embedding`.
 */
template<typename Index_, typename Input_, typename Float_>
Status<Index_, Float_> initialize(const knncolle::Prebuilt<Index_, Input_, Float_>& prebuilt, const std::size_t num_dim, Float_* const embedding, Options options) { 
    auto output = knncolle::find_nearest_neighbors(prebuilt, options.num_neighbors, options.num_threads);
    return initialize(std::move(output), num_dim, embedding, std::move(options));
}

/**
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the input data, neighbor distances and output embedding.
 * @tparam Matrix_ Class of the input matrix for the neighbor search.
 * This should be a `knncolle::SimpleMatrix` or its base class (i.e., `knncolle::Matrix`).
 * 
 * @param data_dim Number of dimensions of the input dataset.
 * @param num_obs Number of observations in the input dataset.
 * @param[in] data Pointer to an array containing the input high-dimensional data as a column-major matrix.
 * Each row corresponds to a dimension (`data_dim`) and each column corresponds to an observation (`num_obs`).
 * @param builder Algorithm to use for the neighbor search.
 * @param num_dim Number of dimensions of the embedding.
 * @param[in, out] embedding Pointer to an array in which to store the embedding.
 * This is treated as a column-major matrix where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * Existing values in this array will be used as input if `Options::initialize = InitializeMethod::NONE`, 
 * and may be used as input if `Options::initialize = InitializeMethod::SPECTRAL` and `Options::initialize_random_on_spectral_fail = false`;
 * otherwise it is only used as output.
 * @param options Further options.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 * Further calls to `Status::run()` will update the embeddings in `embedding`.
 */
template<typename Index_, typename Float_, class Matrix_ = knncolle::Matrix<Index_, Float_> >
Status<Index_, Float_> initialize(
    const std::size_t data_dim,
    const Index_ num_obs,
    const Float_* const data,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    const std::size_t num_dim,
    Float_* const embedding,
    Options options)
{ 
    const auto prebuilt = builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(data_dim, num_obs, data));
    return initialize(*prebuilt, num_dim, embedding, std::move(options));
}

}

#endif
