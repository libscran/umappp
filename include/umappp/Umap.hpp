#ifndef UMAPPP_UMAP_HPP
#define UMAPPP_UMAP_HPP

#include "NeighborList.hpp"
#include "combine_neighbor_sets.hpp"
#include "find_ab.hpp"
#include "neighbor_similarities.hpp"
#include "optimize_layout.hpp"
#include "spectral_init.hpp"

#ifndef UMAPPP_CUSTOM_NEIGHBORS
#include "knncolle/knncolle.hpp"
#endif

#include <random>
#include <cstdint>

/**
 * @file Umap.hpp
 *
 * @brief Run the UMAP algorithm.
 */

namespace umappp {

/**
 * How should the initial coordinates of the embedding be obtained?
 *
 * - `SPECTRAL`: attempts initialization based on spectral decomposition of the graph Laplacian.
 * If that fails, we fall back to random draws from a normal distribution.
 * - `SPECTRAL_ONLY`: attempts spectral initialization as before,
 * but if that fails, we use the existing values in the supplied embedding array.
 * - `RANDOM`: fills the embedding with random draws from a normal distribution.
 * - `NONE`: uses the existing values in the supplied embedding array.
 */
enum InitMethod { SPECTRAL, SPECTRAL_ONLY, RANDOM, NONE };

/**
 * @cond
 */
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
/**
 * @endcond
 */

/**
 * @brief Wrapper class to run UMAP.
 *
 * The Uniform Manifold Approximation and Projection (UMAP) algorithm is an efficient dimensionality reduction method based on nearest neighbors.
 * The general idea is to find a low-dimensional embedding that preserves the neighborhood of each observation from the original space;
 * this is achieved by applying attractive forces between each observation and its neighbors while repelling to all other cells.
 * Further theoretical details can be found in the [original UMAP documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html),
 * though this particular implementation is derived from the C++ code in the [**uwot** R package](https://github.com/jlmelville/uwot).
 *
 * @tparam Float Floating-point type.
 * Defaults to `double` to be conservative, but most applications can make do with `float` for some extra speed.
 *
 * @see
 * McInnes L, Healy J and Melville J (2020).
 * UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
 * _arXiv_, https://arxiv.org/abs/1802.03426
 */
template<typename Float = double>
class Umap {
public:
    /**
     * @brief Default values for all UMAP parameters.
     */
    struct Defaults {
        /**
         * See `set_local_connectivity()`.
         */
        static constexpr Float local_connectivity = 1.0;

        /**
         * See `set_bandwidth()`.
         */
        static constexpr Float bandwidth = 1;

        /**
         * See `set_mix_ratio()`.
         */
        static constexpr Float mix_ratio = 1;

        /**
         * See `set_spread()`.
         */
        static constexpr Float spread = 1;

        /**
         * See `set_min_dist()`.
         */
        static constexpr Float min_dist = 0.01;

        /**
         * See `set_a()`.
         */
        static constexpr Float a = 0;

        /**
         * See `set_b()`.
         */
        static constexpr Float b = 0;

        /**
         * See `set_repulsion_strength()`.
         */
        static constexpr Float repulsion_strength = 1;

        /**
         * See `set_initialize()`.
         */
        static constexpr InitMethod initialize = SPECTRAL;

        /**
         * See `set_num_epochs()`.
         */
        static constexpr int num_epochs = -1;

        /**
         * See `set_learning_rate()`.
         */
        static constexpr Float learning_rate = 1; 

        /**
         * See `set_negative_sample_rate()`.
         */
        static constexpr Float negative_sample_rate = 5;

        /**
         * See `set_num_neighbors()`.
         */
        static constexpr int num_neighbors = 15;

        /**
         * See `set_seed()`.
         */
        static constexpr uint64_t seed = 1234567890;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_parallel_optimization()`.
         */
        static constexpr int parallel_optimization = false;
    };

private:
    InitMethod init = Defaults::initialize;
    int num_neighbors = Defaults::num_neighbors;
    Float local_connectivity = Defaults::local_connectivity;
    Float bandwidth = Defaults::bandwidth;
    Float mix_ratio = Defaults::mix_ratio;
    Float spread = Defaults::spread;
    Float min_dist = Defaults::min_dist;
    int num_epochs = Defaults::num_epochs;
    Float negative_sample_rate = Defaults::negative_sample_rate;
    uint64_t seed = Defaults::seed;

    struct RuntimeParameters {
        Float a = Defaults::a;
        Float b = Defaults::b;
        Float repulsion_strength = Defaults::repulsion_strength;
        Float learning_rate = Defaults::learning_rate;
        int nthreads = Defaults::num_threads;
        bool parallel_optimization = Defaults::parallel_optimization;
    };

    RuntimeParameters rparams;

public:
    /** 
     * @param i How to initialize the embedding, see `InitMethod` for more details.
     * Some choices may use the existing coordinates provided to `run()` via the `embedding` argument.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_initialize(InitMethod i = Defaults::initialize) {
        init = i;
        return *this;
    }

    /**
     * @param n Number of neighbors to use to define the fuzzy sets.
     * Larger values improve connectivity and favor preservation of global structure, at the cost of increased computational work.
     * This argument is only used in certain `run()` and `initialize()` methods that perform identification of the nearest neighbors. 
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_num_neighbors(Float n = Defaults::num_neighbors) {
        num_neighbors = n;
        return *this;
    }

    /**
     * @param l The number of nearest neighbors that are assumed to be always connected, with maximum membership confidence.
     * Larger values increase the connectivity of the embedding and reduce the focus on local structure.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_local_connectivity(Float l = Defaults::local_connectivity) {
        local_connectivity = l;
        return *this;
    }

    /**
     * @param b Effective bandwidth of the kernel when converting the distance to a neighbor into a fuzzy set membership confidence.
     * Larger values reduce the decay in confidence with respect to distance, increasing connectivity and favoring global structure. 
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_bandwidth(Float b = Defaults::bandwidth) {
        bandwidth = b;
        return *this;
    }

    /**
     * @param m Mixing ratio to use when combining fuzzy sets.
     * This symmetrizes the sets by ensuring that the confidence of $A$ belonging to $B$'s set is the same as the confidence of $B$ belonging to $A$'s set.
     * A mixing ratio of 1 will take the union of confidences, a ratio of 0 will take the intersection, and intermediate values will interpolate between them.
     * Larger values (up to 1) favor connectivity and more global structure.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_mix_ratio(Float m = Defaults::mix_ratio) {
        mix_ratio = m;
        return *this;
    }

    /**
     * @param s Scale of the coordinates of the final low-dimensional embedding.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_spread(Float s = Defaults::spread) {
        spread = s;
        return *this;
    }

    /**
     * @param m Minimum distance between observations in the final low-dimensional embedding.
     * Smaller values will increase local clustering while larger values favors a more even distribution.
     * This is interpreted relative to the spread of points in `set_spread()`.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_min_dist(Float m = Defaults::min_dist) {
        min_dist = m;
        return *this;
    }

    /**
     * @param n Number of epochs for the gradient descent, i.e., optimization iterations. 
     *
     * Larger values improve accuracy at the cost of computational work.
     * If the requested number of epochs is negative, a value is automatically chosen based on the size of the dataset:
     *
     * - For datasets with no more than 10000 observations, the number of epochs is set to 500.
     * - For larger datasets, the number of epochs decreases from 500 according to the number of cells beyond 10000, to a lower limit of 200.
     *
     * This choice aims to reduce computational work for very large datasets. 
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_num_epochs(int n = Defaults::num_epochs) {
        num_epochs = n;
        return *this;
    }

    /**
     * @param n Rate of sampling negative observations to compute repulsive forces.
     * This is interpreted with respect to the number of neighbors with attractive forces, i.e., for each attractive interaction, `n` negative samples are taken for repulsive interactions.
     * Smaller values can improve the speed of convergence but at the cost of stability.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_negative_sample_rate(Float n = Defaults::negative_sample_rate) {
        negative_sample_rate = n;
        return *this;
    }

    /**
     * @param s Seed to use for the Mersenne Twister when sampling negative observations.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_seed(uint64_t s = Defaults::seed) {
        seed = s;
        return *this;
    }

    /**
     * @param a Positive value for the $a$ parameter for the fuzzy set membership strength calculations.
     * Larger values yield a sharper decay in membership strength with increasing distance between observations.
     *
     * If this or `set_b()` is set to zero, a suitable value for this parameter is automatically determined from the values provided to `set_spread()` and `set_min_dist()`.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_a(Float a = Defaults::a) {
        rparams.a = a;
        return *this;
    }

    /**
     * @param b Value in $(0, 1)$ for the $b$ parameter for the fuzzy set membership strength calculations.
     * Larger values yield an earlier decay in membership strength with increasing distance between observations.
     *
     * If this or `set_a()` is set to zero, a suitable value for this parameter is automatically determined from the values provided to `set_spread()` and `set_min_dist()`.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_b(Float b = Defaults::b) {
        rparams.b = b;
        return *this;
    }

    /** 
     * @param r Modifier for the repulsive force.
     * Larger values increase repulsion and favor local structure.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_repulsion_strength(Float r = Defaults::repulsion_strength) {
        rparams.repulsion_strength = r;
        return *this;
    }

    /**
     * @param l Initial learning rate used in the gradient descent.
     * Larger values can improve the speed of convergence but at the cost of stability.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_learning_rate(Float l = Defaults::learning_rate) {
        rparams.learning_rate = l;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     *
     * @return A reference to this `Umap` object.
     *
     * This setting affects nearest neighbor detection (if an existing list of neighbors is not supplied in `initialize()` or `run()`) and spectral initialization.
     * If `set_parallel_optimization()` is true, it will also affect the layout optimization, i.e., the gradient descent iterations.
     *
     * The `UMAPPP_CUSTOM_PARALLEL` macro can be set to a function that specifies a custom parallelization scheme.
     * This function should be a template that accept three arguments:
     *
     * - `njobs`, an integer specifying the number of jobs.
     * - `fun`, a lambda that accepts two arguments, `start` and `end`.
     * - `nthreads`, an integer specifying the number of threads to use.
     *
     * The function should split `[0, njobs)` into any number of contiguous, non-overlapping intervals, and call `fun` on each interval, possibly in different threads.
     * The details of the splitting and evaluation are left to the discretion of the developer defining the macro. 
     * The function should only return once all evaluations of `fun` are complete.
     *
     * If `UMAPPP_CUSTOM_PARALLEL` is set, the `IRLBA_CUSTOM_PARALLEL` macro is also set if it is not already defined.
     * This ensures that any custom parallelization scheme is propagated to all of **umappp**'s dependencies.
     * If **irlba** is used outside of **umappp**, some care is required to ensure that the macros are consistently defined throughout the client library/application;
     * otherwise, developers may observe ODR compilation errors. 
     */
    Umap& set_num_threads(int n = Defaults::num_threads) {
        rparams.nthreads = n;
        return *this;
    }

    /**
     * @param p Whether to enable parallel optimization.
     * If set to `true`, this will use the number of threads specified in `set_num_threads()` for the layout optimization step.
     *
     * @return A reference to this `Umap` object.
     *
     * By default, this is set to `false` as the increase in the number of threads is usually not cost-effective for layout optimization.
     * Specifically, while CPU usage scales with the number of threads, the time spent does not decrease by the same factor.
     * We also expect that the number of available CPUs is at least equal to the requested number of threads, otherwise contention will greatly degrade performance.
     * Nonetheless, users can enable parallel optimization if cost is no issue - usually a higher number of threads (above 4) is required to see a reduction in time.
     *
     * If the `UMAPPP_NO_PARALLEL_OPTIMIZATION` macro is defined, **umappp** will not be compiled with support for parallel optimization.
     * This may be desirable in environments that have no support for threading or atomics, or to reduce the binary size if parallelization is not of interest.
     * In such cases, enabling parallel optimization and calling `Status::run()` will raise an error.
     */
    Umap& set_parallel_optimization(bool p = Defaults::parallel_optimization) {
        rparams.parallel_optimization = p;
        return *this;
    }

public:
    /**
     * @brief Status of the UMAP optimization iterations.
     */
    struct Status {
        /**
         * @cond
         */
        Status(EpochData<Float> e, uint64_t seed, RuntimeParameters p, int n, Float* embed) : 
            epochs(std::move(e)), engine(seed), rparams(std::move(p)), ndim_(n), embedding_(embed) {}

        EpochData<Float> epochs;
        std::mt19937_64 engine;
        RuntimeParameters rparams;
        int ndim_;
        Float* embedding_;
        /**
         * @endcond
         */

        /**
         * @return Number of dimensions of the embedding.
         */
        int ndim() const {
            return ndim_;
        }

        /**
         * @return Pointer to a two-dimensional column-major array where rows are dimensions (`ndim`) and columns are observations.
         * This is updated by `initialize()` to store the final embedding.
         */
        const Float* embedding() const {
            return embedding_;
        }

        /**
         * @return Current epoch.
         */
        int epoch() const {
            return epochs.current_epoch;
        }

        /**
         * @return Total number of epochs.
         * This is equal to the value set by `set_num_epochs()` when the `Status` object is created.
         */
        int num_epochs() const {
            return epochs.total_epochs;
        }

        /**
         * @return The number of observations in the dataset.
         */
        size_t nobs() const {
            return epochs.head.size();
        }

        /** 
         * The status of the algorithm and the coordinates in `embedding()` are updated to the specified number of epochs. 
         *
         * @param epoch_limit Number of epochs to run to.
         * The actual number of epochs performed is equal to the difference between `epoch_limit` and the current number of epochs in `epoch()`.
         * `epoch_limit` should be not less than `epoch()` and be no greater than the maximum number of epochs specified in `Umap::set_num_epochs()`.
         * If zero, defaults to the maximum number of epochs. 
         *
         */
        void run(int epoch_limit = 0) {
            if (rparams.nthreads == 1 || !rparams.parallel_optimization) {
                optimize_layout(
                    ndim_,
                    embedding_,
                    epochs,
                    rparams.a,
                    rparams.b,
                    rparams.repulsion_strength,
                    rparams.learning_rate,
                    engine,
                    epoch_limit
                );
            } else {
                optimize_layout_parallel(
                    ndim_,
                    embedding_,
                    epochs,
                    rparams.a,
                    rparams.b,
                    rparams.repulsion_strength,
                    rparams.learning_rate,
                    engine,
                    epoch_limit,
                    rparams.nthreads
                );
            }
            return;
        }
    };

    /** 
     * @param x Indices and distances to the nearest neighbors for each observation.
     * Note the expectations in the `NeighborList` documentation.
     * @param ndim Number of dimensions of the embedding.
     * @param[in, out] embedding Two-dimensional array to store the embedding, 
     * where rows are dimensions (`ndim`) and columns are observations (`x.size()`).
     *
     * @return A `Status` object containing the initial state of the UMAP algorithm, to be used in `run()`.
     * If `set_initialize()` is `NONE` or if spectral initialization fails with `SPECTRAL_ONLY`, `embedding` should contain the initial coordinates and will not be altered;
     * otherwise, it is filled with initial coordinates.
     */
    Status initialize(NeighborList<Float> x, int ndim, Float* embedding) const {
        neighbor_similarities(x, local_connectivity, bandwidth);
        combine_neighbor_sets(x, mix_ratio);

        // Choosing the manner of initialization.
        if (init == SPECTRAL || init == SPECTRAL_ONLY) {
            bool attempt = spectral_init(x, ndim, embedding, rparams.nthreads);
            if (!attempt && init == SPECTRAL) {
                random_init(x.size(), ndim, embedding);
            }
        } else if (init == RANDOM) {
            random_init(x.size(), ndim, embedding);
        }

        // Finding a good a/b pair.
        auto pcopy = rparams;
        if (pcopy.a <= 0 || pcopy.b <= 0) {
            auto found = find_ab(spread, min_dist);
            pcopy.a = found.first;
            pcopy.b = found.second;
        }

        int num_epochs_to_do = choose_num_epochs(num_epochs, x.size());

        return Status(
            similarities_to_epochs(x, num_epochs_to_do, negative_sample_rate),
            seed,
            std::move(pcopy),
            ndim,
            embedding
        );
    }

public:
    /**
     * @tparam Algorithm `knncolle::Base` subclass implementing a nearest neighbor search algorithm.
     * 
     * @param searcher Pointer to a `knncolle::Base` subclass with a `find_nearest_neighbors()` method.
     * @param ndim Number of dimensions of the embedding.
     * @param[out] embedding Two-dimensional array to store the embedding, 
     * where rows are dimensions (`ndim`) and columns are observations (`searcher->nobs()`).
     *
     * @return A `Status` object containing the initial state of the UMAP algorithm, to be used in `run()`.
     * If `set_initialize()` is true, `embedding` is filled with initial coordinates derived from the fuzzy set graph;
     * otherwise it is ignored.
     */
    template<class Algorithm>
    Status initialize(const Algorithm* searcher, int ndim, Float* embedding) { 
        const size_t N = searcher->nobs();
        NeighborList<Float> output(N);

#ifndef UMAPPP_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(rparams.nthreads)
        for (size_t i = 0; i < N; ++i) {
#else
        UMAPPP_CUSTOM_PARALLEL(N, [&](size_t first, size_t last) -> void {
        for (size_t i = first; i < last; ++i) {
#endif

            output[i] = searcher->find_nearest_neighbors(i, num_neighbors);

#ifndef UMAPPP_CUSTOM_PARALLEL
        }
#else
        }
        }, rparams.nthreads);
#endif

        return initialize(std::move(output), ndim, embedding);
    }

#ifndef UMAPPP_CUSTOM_NEIGHBORS
    /**
     * @tparam Input Floating point type for the input data.
     * 
     * @param ndim_in Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] input Pointer to a 2D array containing the input high-dimensional data, with number of rows and columns equal to `ndim_in` and `nobs`, respectively.
     * The array is treated as column-major where each row corresponds to a dimension and each column corresponds to an observation.
     * @param ndim_out Number of dimensions of the embedding.
     * @param[out] embedding Two-dimensional array to store the embedding, 
     * where rows are dimensions (`ndim`) and columns are observations (`searcher->nobs()`).
     *
     * @return A `Status` object containing various pre-computed structures required for the epochs in `run()`.
     * If `set_initialize()` is `NONE` or if spectral initialization fails with `SPECTRAL_ONLY`, `embedding` should contain the initial coordinates and will not be altered;
     * otherwise, it is filled with initial coordinates.
     *
     * This differs from the other `initialize()` methods in that it will internally compute the nearest neighbors for each observation.
     * It will use vantage point trees for the search - see the other `initialize()` methods to specify a custom search algorithm.
     */
    template<typename Input = Float>
    Status initialize(int ndim_in, size_t nobs, const Input* input, int ndim_out, Float* embedding) { 
        knncolle::VpTreeEuclidean<> searcher(ndim_in, nobs, input); 
        return initialize(&searcher, ndim_out, embedding);
    }
#endif

public:
    /**
     * @tparam Algorithm `knncolle::Base` subclass implementing a nearest neighbor search algorithm.
     * 
     * @param searcher Pointer to a `knncolle::Base` subclass with a `find_nearest_neighbors()` method.
     * @param ndim Number of dimensions of the embedding.
     * @param[in, out] embedding Two-dimensional array where rows are dimensions (`ndim`) and columns are observations (`searcher->nobs()`).
     * This is filled with the final embedding on output.
     * If `set_initialize()` is false, this is assumed to contain the initial coordinates on input.
     * @param epoch_limit Number of epochs to run - see `Status::run()`.
     *
     * @return The status of the algorithm is returned after running up to `epoch_limit`; this can be used for further iterations by invoking `Status::run()`.
     * `embedding` is updated with the embedding at the specified epoch limit.
     */
    template<class Algorithm> 
    Status run(const Algorithm* searcher, int ndim, Float* embedding, int epoch_limit = 0) {
        auto status = initialize(searcher, ndim, embedding);
        status.run(epoch_limit);
        return status;
    }

    /** 
     * @param x Indices and distances to the nearest neighbors for each observation.
     * Note the expectations in the `NeighborList` documentation.
     * @param ndim Number of dimensions of the embedding.
     * @param[in, out] embedding Two-dimensional array where rows are dimensions (`ndim`) and columns are observations.
     * This is filled with the final embedding on output.
     * If `set_initialize()` is `NONE` or if spectral initialization fails with `SPECTRAL_ONLY`, `embedding` is assumed to contain the initial coordinates on input.
     * @param epoch_limit Number of epochs to run - see `Status::run()`.
     *
     * @return The status of the algorithm is returned after running up to `epoch_limit`; this can be used for further iterations by invoking `Status::run()`.
     * `embedding` is updated with the embedding at the specified epoch limit.
     */
    Status run(NeighborList<Float> x, int ndim, Float* embedding, int epoch_limit = 0) const {
        auto status = initialize(std::move(x), ndim, embedding);
        status.run(epoch_limit);
        return status;
    }

#ifndef UMAPPP_CUSTOM_NEIGHBORS
    /**
     * @tparam Input Floating point type for the input data.
     * 
     * @param ndim_in Number of dimensions.
     * @param nobs Number of observations.
     * @param[in] input Pointer to a 2D array containing the input high-dimensional data, with number of rows and columns equal to `ndim_in` and `nobs`, respectively.
     * The array is treated as column-major where each row corresponds to a dimension and each column corresponds to an observation.
     * @param ndim_out Number of dimensions of the embedding.
     * @param[in, out] embedding Two-dimensional array where rows are dimensions (`ndim`) and columns are observations (`searcher->nobs()`).
     * This is filled with the final embedding on output.
     * If `set_initialize()` is `NONE` or if spectral initialization fails with `SPECTRAL_ONLY`, `embedding` is assumed to contain the initial coordinates on input.
     * @param epoch_limit Number of epochs to run - see `Status::run()`.
     *
     * @return The status of the algorithm is returned after running up to `epoch_limit`; this can be used for further iterations by invoking `Status::run()`.
     * `embedding` is updated with the embedding at the specified epoch limit.
     */
    template<typename Input = Float>
    Status run(int ndim_in, size_t nobs, const Input* input, int ndim_out, Float* embedding, int epoch_limit = 0) {
        auto status = initialize(ndim_in, nobs, input, ndim_out, embedding);
        status.run(epoch_limit);
        return status;
    }
#endif
};

}

#endif
