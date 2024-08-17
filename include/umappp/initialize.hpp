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
#include <cstdint>

/**
 * @file initialize.hpp
 * @brief Initialize the UMAP algorithm.
 */

namespace umappp {

/**
 * @cond
 */
namespace internal {

inline int choose_num_epochs(int num_epochs, size_t size) {
    if (num_epochs < 0) {
        // Choosing the number of epochs. We use a simple formula to decrease
        // the number of epochs with increasing size, with the aim being that
        // the 'extra work' beyond the minimal 200 epochs should be the same
        // regardless of the numbe of observations. Given one calculation per
        // observation per epoch, this amounts to 300 * 10000 calculations at
        // the lower bound, so we simply choose a number of epochs that
        // equalizes the number of calculations for any number of observations.
        if (num_epochs < 0) {
            constexpr int limit = 10000, minimal = 200, maximal = 300;
            if (size <= limit) {
                num_epochs = minimal + maximal;
            } else {
                num_epochs = minimal + static_cast<int>(std::ceil(maximal * limit / static_cast<double>(size)));
            }
        }
    }
    return num_epochs;
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
 * @param[in, out] embedding Pointer to an array in which to store the embedding, where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * This is only used as input if `Options::init == InitializeMethod::NONE`, otherwise it is only used as output.
 * The lifetime of the array should be no shorter than the final call to `Status::run()`.
 * @param options Further options.
 * Note that `Options::num_neighbors` is ignored here.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 * Further calls to `Status::run()` will update the embeddings in `embedding`.
 */
template<typename Index_, typename Float_>
Status<Index_, Float_> initialize(NeighborList<Index_, Float_> x, int num_dim, Float_* embedding, Options options) {
    internal::neighbor_similarities<Index_, Float_>(x, options.local_connectivity, options.bandwidth);
    internal::combine_neighbor_sets<Index_, Float_>(x, options.mix_ratio);

    // Choosing the manner of initialization.
    if (options.initialize == InitializeMethod::SPECTRAL || options.initialize == InitializeMethod::SPECTRAL_ONLY) {
        bool attempt = internal::spectral_init(x, num_dim, embedding, options.num_threads);
        if (!attempt && options.initialize == InitializeMethod::SPECTRAL) {
            internal::random_init(x.size(), num_dim, embedding);
        }
    } else if (options.initialize == InitializeMethod::RANDOM) {
        internal::random_init(x.size(), num_dim, embedding);
    }

    // Finding a good a/b pair.
    if (options.a <= 0 || options.b <= 0) {
        auto found = internal::find_ab(options.spread, options.min_dist);
        options.a = found.first;
        options.b = found.second;
    }

    options.num_epochs = internal::choose_num_epochs(options.num_epochs, x.size());

    return Status<Index_, Float_>(
        internal::similarities_to_epochs<Index_, Float_>(x, options.num_epochs, options.negative_sample_rate),
        options,
        num_dim,
        embedding
    );
}

/**
 * @tparam Dim_ Integer type for the dimensions of the input dataset.
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Float_ Floating-point type for the distances.
 *
 * @param prebuilt A `knncolle::Prebuilt` instance constructed from the input dataset.
 * @param num_dim Number of dimensions of the UMAP embedding.
 * @param[in, out] embedding Pointer to an array in which to store the embedding, where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * This is only used as input if `Options::init == InitializeMethod::NONE`, otherwise it is only used as output.
 * The lifetime of the array should be no shorter than the final call to `Status::run()`.
 * @param options Further options.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 * Further calls to `Status::run()` will update the embeddings in `embedding`.
 */
template<typename Dim_, typename Index_, typename Float_>
Status<Index_, Float_> initialize(const knncolle::Prebuilt<Dim_, Index_, Float_>& prebuilt, int num_dim, Float_* embedding, Options options) { 
    const size_t N = prebuilt.num_observations();
    NeighborList<Index_, Float_> output(N);

#ifndef UMAPPP_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(options.num_threads)
#endif
    {
#else
    UMAPPP_CUSTOM_PARALLEL(N, [&](size_t first, size_t last) -> void {
#endif

        auto searcher = prebuilt.initialize();
        std::vector<Index_> indices;
        std::vector<Float_> distances;

#ifndef UMAPPP_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp for
#endif
        for (size_t i = 0; i < N; ++i) {
#else
        for (size_t i = first; i < last; ++i) {
#endif

            searcher->search(i, options.num_neighbors, &indices, &distances);
            size_t actual_k = indices.size(); 
            for (size_t x = 0; x < actual_k; ++x) {
                output[i].emplace_back(indices[x], distances[x]);
            }

#ifndef UMAPPP_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, options.num_threads);
#endif

    return initialize(std::move(output), num_dim, embedding, std::move(options));
}

/**
 * @tparam Dim_ Integer type for the dimensions of the input dataset.
 * @tparam Index_ Integer type of the neighbor indices.
 * @tparam Float_ Floating-point type for the distances.
 * 
 * @param data_dim Number of dimensions of the input dataset.
 * @param num_obs Number of observations in the input dataset.
 * @param[in] data Pointer to an array containing the input high-dimensional data as a column-major matrix.
 * Each row corresponds to a dimension (`data_dim`) and each column corresponds to an observation (`num_obs`).
 * @param builder Algorithm to use for the neighbor search.
 * @param num_dim Number of dimensions of the embedding.
 * @param[in, out] embedding Pointer to an array in which to store the embedding, where rows are dimensions (`num_dim`) and columns are observations (`x.size()`).
 * This is only used as input if `Options::init == InitializeMethod::NONE`, otherwise it is only used as output.
 * The lifetime of the array should be no shorter than the final call to `Status::run()`.
 * @param options Further options.
 *
 * @return A `Status` object containing the initial state of the UMAP algorithm.
 * Further calls to `Status::run()` will update the embeddings in `embedding`.
 */
template<typename Dim_, typename Index_, typename Float_>
Status<Index_, Float_> initialize(
    Dim_ data_dim,
    Index_ num_obs,
    const Float_* data,
    const knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_>& builder,
    int num_dim,
    Float_* embedding,
    Options options)
{ 
    auto prebuilt = builder.build_unique(knncolle::SimpleMatrix<Dim_, Index_, Float_>(data_dim, num_obs, data));
    return initialize(*prebuilt, num_dim, embedding, std::move(options));
}

}

#endif
