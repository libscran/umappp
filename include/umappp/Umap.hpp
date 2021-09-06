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

namespace umappp {

class Umap {
public:
    struct Defaults {
        static constexpr double local_connectivity = 1.0;

        static constexpr double bandwidth = 1;

        static constexpr double mix_ratio = 1;

        static constexpr double spread = 1;

        static constexpr double min_dist = 0.01;

        static constexpr double a = 0;

        static constexpr double b = 0;

        static constexpr double gamma = 1;

        static constexpr bool init = true;

        static constexpr int num_epochs = 500;

        static constexpr double learning_rate = 1; 

        static constexpr double negative_sample_rate = 5;

        static constexpr int num_neighbors = 15;

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

    double gamma = Defaults::gamma;

    bool init = Defaults::init;

    int num_epochs = Defaults::num_epochs;

    double learning_rate = Defaults::learning_rate;

    double negative_sample_rate = Defaults::negative_sample_rate;

    int num_neighbors = Defaults::num_neighbors;

    uint64_t seed = Defaults::seed;

public:
    Umap& set_a(double a_ = Defaults::a) {
        a = a_;
        return *this;
    }

    Umap& set_b(double b_ = Defaults::b) {
        b = b_;
        return *this;
    }

public:
    struct Status {
        Status(EpochData e, uint64_t seed, double a_, double b_) : epochs(std::move(e)), engine(seed), a(a_), b(b_) {}
        EpochData epochs;
        std::mt19937_64 engine;
        double a;
        double b;

        int epoch() const {
            return epochs.current_epoch;
        }
    };

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
    void run(Status& s, int ndim, double* embedding, int epoch_limit = 0) const {
        optimize_layout(
            ndim,
            embedding,
            s.epochs,
            s.a,
            s.b,
            gamma,
            learning_rate,
            s.engine,
            epoch_limit
        );
        return;
    }

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
     * @param[in] input Pointer to a 2D array containing the input high-dimensional data, with number of rows and columns equal to `D` and `N`, respectively.
     * The array is treated as column-major where each row corresponds to a dimension and each column corresponds to an observation.
     * @param D Number of dimensions.
     * @param N Number of observations.
     *
     * @return A `Status` object containing various pre-computed structures required for the iterations in `run()`.
     *
     * This differs from the other `run()` methods in that it will internally compute the nearest neighbors for each observation.
     * As with the original t-SNE implementation, it will use vantage point trees for the search.
     * See the other `initialize()` methods to specify a custom search algorithm.
     */
    template<typename Input = double>
    auto initialize(int ndim_in, size_t nobs, const Input* input, int ndim_out, double* embedding) { 
#ifdef PROGRESS_PRINTER
        PROGRESS_PRINTER("umappp::Umap::initialize", "Constructing neighbor search indices")
#endif
        knncolle::VpTreeEuclidean<> searcher(ndim_in, nobs, input); 
        return initialize(&searcher, ndim_out, embedding);
    }

    /**
     * @tparam Input Floating point type for the input data.
     * 
     * @param[in] input Pointer to a 2D array containing the input high-dimensional data, with number of rows and columns equal to `D` and `N`, respectively.
     * The array is treated as column-major where each row corresponds to a dimension and each column corresponds to an observation.
     * @param D Number of dimensions.
     * @param N Number of observations.
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn_index.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial locations of each observation; on output, it is updated to the final t-SNE locations.
     *
     * @return A `Status` object containing the final state of the algorithm after applying all iterations.
     */
    template<typename Input = double>
    auto run(int ndim_in, size_t nobs, const Input* input, int ndim_out, double* embedding) {
        auto status = initialize(ndim_in, nobs, input, ndim_out, embedding);
        run(status, ndim_out, embedding);
        return status;
    }
#endif

    /**
     * @tparam Algorithm `knncolle::Base` subclass implementing a nearest neighbor search algorithm.
     * 
     * @param searcher Pointer to a `knncolle::Base` subclass with a `find_nearest_neighbors()` method.
     *
     * @return A `Status` object containing various pre-computed structures required for the iterations in `run()`.
     *
     * Compared to other `initialize()` methods, this provides more fine-tuned control over the nearest neighbor search parameters.
     */
    template<class Algorithm>
    auto initialize(const Algorithm* searcher, int ndim, double* embedding) { 
        NeighborList output;
        const size_t N = searcher->nobs();
        output.reserve(N);

#ifdef PROGRESS_PRINTER
        PROGRESS_PRINTER("qdtsne::Tsne::initialize", "Searching for nearest neighbors")
#endif
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            auto out = searcher->find_nearest_neighbors(i, num_neighbors);
            out.emplace_back(i, 0);
            output.emplace_back(std::move(out));
        }

        return initialize(output, ndim, embedding);
    }

    /**
     * @tparam Algorithm `knncolle::Base` subclass implementing a nearest neighbor search algorithm.
     * 
     * @param searcher Pointer to a `knncolle::Base` subclass with a `find_nearest_neighbors()` method.
     * @param[in, out] Y Pointer to a 2D array with number of rows and columns equal to `ndim` and `nn_index.size()`, respectively.
     * The array is treated as column-major where each column corresponds to an observation.
     * On input, this should contain the initial locations of each observation; on output, it is updated to the final t-SNE locations.
     *
     * @return A `Status` object containing the final state of the algorithm after applying all iterations.
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
