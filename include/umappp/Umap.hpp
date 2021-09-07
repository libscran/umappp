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
 * @brief Wrapper class to run UMAP.
 *
 * The Uniform Manifold Approximation and Projection (UMAP) algorithm is an efficient dimensionality reduction method based on nearest neighbors.
 * The general idea is to find a low-dimensional embedding that preserves the neighborhood of each observation from the original space;
 * this is achieved by applying attractive forces between each observation and its neighbors while repelling to all other cells.
 * Further theoretical details can be found in the [original UMAP documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html),
 * though this particular implementation is derived from the C++ code in the [**uwot** R package](https://github.com/jlmelville/uwot).
 *
 * @see
 * McInnes L and Healy J (2018).
 * UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction 
 * _arXiv_, https://arxiv.org/abs/1802.03426
 */
class Umap {
public:
    /**
     * @brief Default values for all UMAP parameters.
     */
    struct Defaults {
        /**
         * See `set_local_connectivity()`.
         */
        static constexpr double local_connectivity = 1.0;

        /**
         * See `set_bandwidth()`.
         */
        static constexpr double bandwidth = 1;

        /**
         * See `set_mix_ratio()`.
         */
        static constexpr double mix_ratio = 1;

        /**
         * See `set_spread()`.
         */
        static constexpr double spread = 1;

        /**
         * See `set_min_dist()`.
         */
        static constexpr double min_dist = 0.01;

        /**
         * See `set_a()`.
         */
        static constexpr double a = 0;

        /**
         * See `set_b()`.
         */
        static constexpr double b = 0;

        /**
         * See `set_repulsion_strength()`.
         */
        static constexpr double repulsion_strength = 1;

        /**
         * See `set_initialize()`.
         */
        static constexpr bool initialize = true;

        /**
         * See `set_num_epochs()`.
         */
        static constexpr int num_epochs = 500;

        /**
         * See `set_learning_rate()`.
         */
        static constexpr double learning_rate = 1; 

        /**
         * See `set_negative_sample_rate()`.
         */
        static constexpr double negative_sample_rate = 5;

        /**
         * See `set_num_neighbors()`.
         */
        static constexpr int num_neighbors = 15;

        /**
         * See `set_seed()`.
         */
        static constexpr uint64_t seed = 1234567890;
    };

private:
    double local_connectivity = Defaults::local_connectivity;

    double bandwidth = Defaults::bandwidth;

    double mix_ratio = Defaults::mix_ratio;

    double spread = Defaults::spread;

    double min_dist = Defaults::min_dist;

    double a = Defaults::a;

    double b = Defaults::b;

    double repulsion_strength = Defaults::repulsion_strength;

    bool init = Defaults::initialize;

    int num_epochs = Defaults::num_epochs;

    double learning_rate = Defaults::learning_rate;

    double negative_sample_rate = Defaults::negative_sample_rate;

    int num_neighbors = Defaults::num_neighbors;

    uint64_t seed = Defaults::seed;

public:
    /**
     * @param l The number of nearest neighbors that are assumed to be always connected, with maximum membership confidence.
     * Larger values increase the connectivity of the embedding and reduce the focus on local structure.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_local_connectivity(double l = Defaults::local_connectivity) {
        local_connectivity = l;
        return *this;
    }

    /**
     * @param b Effective bandwidth of the kernel when converting the distance to a neighbor into a fuzzy set membership confidence.
     * Larger values reduce the decay in confidence with respect to distance, increasing connectivity and favoring global structure. 
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_bandwidth(double b = Defaults::bandwidth) {
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
    Umap& set_mix_ratio(double m = Defaults::mix_ratio) {
        mix_ratio = m;
        return *this;
    }

    /**
     * @param s Scale of the coordinates of the final low-dimensional embedding.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_spread(double s = Defaults::spread) {
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
    Umap& set_min_dist(double m = Defaults::min_dist) {
        min_dist = m;
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
    Umap& set_a(double a = Defaults::a) {
        this->a = a;
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
    Umap& set_b(double b = Defaults::b) {
        this->b = b;
        return *this;
    }

    /** 
     * @param r Modifier for the repulsive force.
     * Larger values increase repulsion and favor local structure.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_repulsion_strength(double r = Defaults::repulsion_strength) {
        repulsion_strength = r;
        return *this;
    }

    /** 
     * @param i Whether to initialize the embedding based on a spectral decomposition of the fuzzy set graph.
     * If false, the existing coordinates provided to `run()` via `embedding` are directly used.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_initialize(bool i = Defaults::initialize) {
        init = i;
        return *this;
    }

    /**
     * @param n Number of epochs for the gradient descent, i.e., optimization iterations.
     * Larger values improve convergence at the cost of computational work.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_num_epochs(int n = Defaults::num_epochs) {
        num_epochs = n;
        return *this;
    }

    /**
     * @param l Initial learning rate used in the gradient descent.
     * Larger values can improve the speed of convergence but at the cost of stability.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_learning_rate(double l = Defaults::learning_rate) {
        learning_rate = l;
        return *this;
    }

    /**
     * @param n Rate of sampling negative observations to compute repulsive forces.
     * This is interpreted with respect to the number of neighbors with attractive forces, i.e., for each attractive interaction, `n` negative samples are taken for repulsive interactions.
     * Smaller values can improve the speed of convergence but at the cost of stability.
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_negative_sample_rate(double n = Defaults::negative_sample_rate) {
        learning_rate = l;
        return *this;
    }

    /**
     * @param n Number of neighbors to use to define the fuzzy sets.
     * Larger values improve connectivity and favor preservation of global structure, at the cost of increased computational work.
     * This argument is only used in certain `run()` and `initialize()` methods that perform identification of the nearest neighbors. 
     *
     * @return A reference to this `Umap` object.
     */
    Umap& set_num_neighbors(double n = Defaults::num_neighbors) {
        num_neighbors = n;
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

public:
    /**
     * @brief Status of the UMAP optimization iterations.
     */
    struct Status {
        /**
         * @cond
         */
        Status(EpochData e, uint64_t seed, double a_, double b_) : epochs(std::move(e)), engine(seed), a(a_), b(b_) {}
        EpochData epochs;
        std::mt19937_64 engine;
        double a;
        double b;
        /**
         * @endcond
         */

        /**
         * @return Current epoch.
         */
        int epoch() const {
            return epochs.current_epoch;
        }
    };

    /** 
     * @param x Indices and distances to the nearest neighbors for each observation.
     * @param ndim Number of dimensions of the embedding.
     * @param[out] embedding Two-dimensional array to store the embedding, 
     * where rows are dimensions (`ndim`) and columns are observations (`x.size()`).
     *
     * @return A `Status` object containing the initial state of the UMAP algorithm, to be used in `run()`.
     * If `set_initialize()` is true, `embedding` is filled with initial coordinates derived from the fuzzy set graph;
     * otherwise it is ignored.
     */
    Status initialize(NeighborList x, int ndim, double* embedding) const {
        neighbor_similarities(x, local_connectivity, bandwidth);
        combine_neighbor_sets(x, mix_ratio);

        // Running spectral initialization.
        if (init) {
            spectral_init(x, ndim, embedding);
        }

        // Finding a good a/b pair.
        double a_ = a;
        double b_ = b;
        if (a_ <= 0 || b_ <= 0) {
            auto found = find_ab(spread, min_dist);
            a_ = found.first;
            b_ = found.second;
        }

        return Status(
            similarities_to_epochs(x, num_epochs, negative_sample_rate),
            seed,
            a_, 
            b_
        );
    }

public:
    /** 
     * @param s The status of the algorithm, typically generated by `initialize()`.
     * @param ndim Number of dimensions of the embedding.
     * @param[in, out] embedding Two-dimensional array where rows are dimensions (`ndim`) and columns are observations.
     * This contains the initial coordinates and is updated to store the final embedding.
     * @param epoch_limit Number of epochs to run.
     * If zero, defaults to the maximum number of epochs in `set_num_epochs()`.
     *
     * @return `s` and `embedding` are updated for the given number of epochs. 
     *
     * The number of epochs that are actually used is determined from the difference between `Status::epoch()` and the smaller of `epoch_limit` and the maximum number of epochs.
     * Setting the epoch limit is helpful for running the iterations one at a time.
     */
    void run(Status& s, int ndim, double* embedding, int epoch_limit = 0) const {
        optimize_layout(
            ndim,
            embedding,
            s.epochs,
            s.a,
            s.b,
            repulsion_strength,
            learning_rate,
            s.engine,
            epoch_limit
        );
        return;
    }

    /** 
     * @param x Indices and distances to the nearest neighbors for each observation.
     * @param ndim Number of dimensions of the embedding.
     * @param[in, out] embedding Two-dimensional array where rows are dimensions (`ndim`) and columns are observations.
     * This is filled with the final embedding on output.
     * If `set_initialize()` is false, this is assumed to contain the initial coordinates on input.
     * @param epoch_limit Number of epochs to run, see `run()`.
     * If zero, defaults to the maximum number of epochs in `set_num_epochs()`.
     *
     * @return The status of the algorithm is returned after running up to `epoch_limit`, for re-use in `run()`.
     * `embedding` is updated with the embedding at that point.
     */
    Status run(NeighborList x, int ndim, double* embedding, int epoch_limit = 0) const {
        auto status = initialize(std::move(x), ndim, embedding);
        run(status, ndim, embedding, epoch_limit);
        return status;
    }
public:
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
     * If `set_initialize()` is true, `embedding` is filled with initial coordinates derived from the fuzzy set graph;
     * otherwise it is ignored.
     *
     * This differs from the other `run()` methods in that it will internally compute the nearest neighbors for each observation.
     * It will use vantage point trees for the search - see the other `initialize()` methods to specify a custom search algorithm.
     */
    template<typename Input = double>
    auto initialize(int ndim_in, size_t nobs, const Input* input, int ndim_out, double* embedding) { 
        knncolle::VpTreeEuclidean<> searcher(ndim_in, nobs, input); 
        return initialize(&searcher, ndim_out, embedding);
    }

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
     * If `set_initialize()` is false, this is assumed to contain the initial coordinates on input.
     * @param epoch_limit Number of epochs to run, see `run()`.
     * If zero, defaults to the maximum number of epochs in `set_num_epochs()`.
     *
     * @return The status of the algorithm is returned after running up to `epoch_limit`, for re-use in `run()`.
     * `embedding` is updated with the embedding at that point.
     */
    template<typename Input = double>
    auto run(int ndim_in, size_t nobs, const Input* input, int ndim_out, double* embedding, int epoch_limit = 0) {
        auto status = initialize(ndim_in, nobs, input, ndim_out, embedding);
        run(status, ndim_out, embedding, epoch_limit);
        return status;
    }
#endif

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
    auto initialize(const Algorithm* searcher, int ndim, double* embedding) { 
        NeighborList output;
        const size_t N = searcher->nobs();
        output.reserve(N);

        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            auto out = searcher->find_nearest_neighbors(i, num_neighbors);
            output.emplace_back(std::move(out));
        }

        return initialize(output, ndim, embedding);
    }

    /**
     * @tparam Algorithm `knncolle::Base` subclass implementing a nearest neighbor search algorithm.
     * 
     * @param searcher Pointer to a `knncolle::Base` subclass with a `find_nearest_neighbors()` method.
     * @param ndim Number of dimensions of the embedding.
     * @param[in, out] embedding Two-dimensional array where rows are dimensions (`ndim`) and columns are observations (`searcher->nobs()`).
     * This is filled with the final embedding on output.
     * If `set_initialize()` is false, this is assumed to contain the initial coordinates on input.
     * @param epoch_limit Number of epochs to run.
     * If zero, defaults to the maximum number of epochs in `set_num_epochs()`.
     *
     * @return A `Status` object containing the state of the algorithm after running epochs up to `epoch_limit`.
     * `embedding` is updated with the embedding at that point.
     */
    template<class Algorithm> 
    auto run(const Algorithm* searcher, int ndim, double* embedding, int epoch_limit = 0) {
        auto status = initialize(searcher, ndim, embedding);
        run(status, ndim, embedding, epoch_limit);
        return status;
    }
};

}

#endif
