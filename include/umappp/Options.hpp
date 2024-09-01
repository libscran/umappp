#ifndef UMAPPP_OPTIONS_HPP
#define UMAPPP_OPTIONS_HPP

#include <cstdint>

/**
 * @file Options.hpp
 * @brief Options for the UMAP algorithm.
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
enum InitializeMethod : char { SPECTRAL, SPECTRAL_ONLY, RANDOM, NONE };

/**
 * @brief Options for `initialize()`.
 */
struct Options {
    /**
     * The number of nearest neighbors that are assumed to be always connected, with maximum membership confidence.
     * Larger values increase the connectivity of the embedding and reduce the focus on local structure.
     * This may be a fractional number of neighbors.
     */
    double local_connectivity = 1;

    /**
     * Effective bandwidth of the kernel when converting the distance to a neighbor into a fuzzy set membership confidence.
     * Larger values reduce the decay in confidence with respect to distance, increasing connectivity and favoring global structure. 
     */
    double bandwidth = 1;

    /**
     * Mixing ratio to use when combining fuzzy sets.
     * This symmetrizes the sets by ensuring that the confidence of point \f$A\f$ belonging to point \f$B\f$'s set is the same as the confidence of \f$B\f$ belonging to \f$A\f$'s set.
     * A mixing ratio of 1 will take the union of confidences, a ratio of 0 will take the intersection, and intermediate values will interpolate between them.
     * Larger values (up to 1) favor connectivity and more global structure.
     */
    double mix_ratio = 1;

    /**
     * Scale of the coordinates of the final low-dimensional embedding.
     */
    double spread = 1;

    /**
     * Minimum distance between observations in the final low-dimensional embedding.
     * Smaller values will increase local clustering while larger values favor a more even distribution of points throughout the low-dimensional space.
     * This is interpreted relative to the spread of points in `Options::spread`.
     */
    double min_dist = 0.1;

    /**
     * Positive value for the \f$a\f$ parameter for the fuzzy set membership strength calculations.
     * Larger values yield a sharper decay in membership strength with increasing distance between observations.
     *
     * If this or `Options::a` is set to zero, a suitable value for this parameter is automatically determined from `Options::spread` and `Options::min_dist`.
     */
    double a = 0;

    /**
     * Value in \f$(0, 1)\f$ for the \f$b\f$ parameter for the fuzzy set membership strength calculations.
     * Larger values yield an earlier decay in membership strength with increasing distance between observations.
     *
     * If this or `Options::a` is set to zero, a suitable value for this parameter is automatically determined from the values provided to `Options::spread` and `Options::min_dist`.
     */
    double b = 0;

    /**
     * Modifier for the repulsive force.
     * Larger values increase repulsion and favor local structure.
     */
    double repulsion_strength = 1;

    /** 
     * How to initialize the embedding.
     * Some choices may use the existing coordinates provided to `initialize()` via the `embedding` argument.
     */
    InitializeMethod initialize = InitializeMethod::SPECTRAL;

    /**
     * Number of epochs for the gradient descent, i.e., optimization iterations. 
     * Larger values improve accuracy at the cost of computational work.
     * If the requested number of epochs is negative, a value is automatically chosen based on the size of the dataset:
     *
     * - For datasets with no more than 10000 observations, the number of epochs is set to 500.
     * - For larger datasets, the number of epochs decreases from 500 according to the number of cells beyond 10000, to a lower limit of 200.
     *
     * This choice aims to reduce computational work for very large datasets. 
     */
    int num_epochs = -1;

    /**
     * Initial learning rate used in the gradient descent.
     * Larger values can improve the speed of convergence but at the cost of stability.
     */
    double learning_rate = 1; 

    /**
     * Rate of sampling negative observations to compute repulsive forces.
     * This is interpreted with respect to the number of neighbors with attractive forces, i.e., for each attractive interaction, `n` negative samples are taken for repulsive interactions.
     * Smaller values can improve the speed of convergence but at the cost of stability.
     */
    double negative_sample_rate = 5;

    /**
     * Number of neighbors to use to define the fuzzy sets.
     * Larger values improve connectivity and favor preservation of global structure, at the cost of increased computational work.
     * This argument is only used in certain `initialize()` overloads that perform identification of the nearest neighbors. 
     */
    int num_neighbors = 15;

    /**
     * Seed to use for the Mersenne Twister when sampling negative observations.
     */
    uint64_t seed = 1234567890;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `parallelize()` for most calculations.
     * The exception is the nearest-neighbor search in some of the `initialize()` overloads, where the scheme is determined by `knncolle::parallelize()` instead.
     * 
     * If `Options::parallel_optimization = true`, this option will also affect the layout optimization, i.e., the gradient descent iterations.
     */
    int num_threads = 1;

    /**
     * Whether to enable parallel optimization.
     * If set to `true`, this will use the number of threads specified in `Options::num_threads` for the layout optimization step.
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
    int parallel_optimization = false;
};

}

#endif
