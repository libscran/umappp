#ifndef UMAPPP_OPTIONS_HPP
#define UMAPPP_OPTIONS_HPP

#include <random>
#include <optional>

#include "sanisizer/sanisizer.hpp"
#include "irlba/irlba.hpp"

/**
 * @file Options.hpp
 * @brief Options for the UMAP algorithm.
 */

namespace umappp {

/**
 * How should the initial coordinates of the embedding be obtained?
 *
 * - `SPECTRAL`: spectral decomposition of the normalized graph Laplacian.
 *   Specifically, the initial coordinates are defined from the eigenvectors corresponding to the smallest non-zero eigenvalues.
 *   This fails in the presence of multiple graph components or if the approximate SVD (via `irlba::compute()`) fails to converge.
 * - `RANDOM`: fills the embedding with random draws from a normal distribution.
 * - `NONE`: uses existing values in the supplied embedding array.
 */
enum InitializeMethod : char { SPECTRAL, RANDOM, NONE };

/**
 * Class of the random number generator used in **umappp**.
 */
typedef std::mt19937_64 RngEngine;

/**
 * @brief Options for `initialize()`.
 */
struct Options {
    /**
     * Number of nearest neighbors that are assumed to be always connected, with maximum membership confidence.
     * Larger values increase the connectivity of the embedding and reduce the focus on local structure.
     * This may be a fractional number of neighbors, in which case interpolation is performed when computing the membership confidence.
     */
    double local_connectivity = 1;

    /**
     * Effective bandwidth of the kernel when converting the distance to a neighbor into a fuzzy set membership confidence.
     * Larger values reduce the decay in confidence with respect to distance, increasing connectivity and favoring global structure. 
     */
    double bandwidth = 1;

    /**
     * Mixing ratio in \f$[0, 1]\f$ when combining fuzzy sets.
     * This symmetrizes the sets so that the confidence of observation \f$A\f$ belonging to observation \f$B\f$'s set is the same as that of \f$B\f$ belonging to \f$A\f$'s set.
     * A mixing ratio of 1 will take the union of confidences, a ratio of 0 will take the intersection, and intermediate values will interpolate between them.
     * Larger values favor connectivity and more global structure.
     */
    double mix_ratio = 1;

    /**
     * Scale of the coordinates of the final low-dimensional embedding.
     * Ignored if both `Options::a` and `Options::b` are provided.
     */
    double spread = 1;

    /**
     * Minimum distance between observations in the final low-dimensional embedding.
     * Smaller values will increase local clustering while larger values favor a more even distribution of observations throughout the low-dimensional space.
     * This is interpreted relative to `Options::spread`.
     * Ignored if both `Options::a` and `Options::b` are provided.
     */
    double min_dist = 0.1;

    /**
     * Positive value for the \f$a\f$ parameter for the fuzzy set membership confidence calculations.
     * Larger values yield a sharper decay in membership confidence with increasing distance between observations.
     *
     * If this or `Options::b` are unset, a suitable value for this parameter is automatically determined from `Options::spread` and `Options::min_dist`.
     */
    std::optional<double> a;

    /**
     * Value in \f$(0, 1)\f$ for the \f$b\f$ parameter for the fuzzy set membership confidence calculations.
     * Larger values yield an earlier decay in membership confidence with increasing distance between observations.
     *
     * If this or `Options::a` are unset, a suitable value for this parameter is automatically determined from `Options::spread` and `Options::min_dist`.
     */
    std::optional<double> b;

    /**
     * Modifier for the repulsive force.
     * Larger values increase repulsion and favor local structure.
     */
    double repulsion_strength = 1;

    /** 
     * How to initialize the embedding.
     */
    InitializeMethod initialize_method = InitializeMethod::SPECTRAL;

    /**
     * Whether to fall back to random sampling from a normal distribution (i.e., same as `InitializeMethod::RANDOM`) if spectral initialization fails.
     * If `false`, any existing values in the input array will be used, i.e., same as `InitializeMethod::NONE`.
     * Only relevant if `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization fails.
     */
    bool initialize_random_on_spectral_fail = true;

    /**
     * Further options to pass to `irlba::compute()` for spectral initialization.
     */
    irlba::Options initialize_spectral_irlba_options;

    /**
     * Maximum absolute magnitude of the coordinates after spectral initialization.
     * All initial coordinates are scaled such that the maximum of their absolute values is equal to `initialize_spectral_scale`.
     * This ensures that outlier observations will not have large absolute distances that may interfere with optimization.
     * Only relevant if `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization does not fail.
     */
    double initialize_spectral_scale = 10;

    /**
     * Whether to jitter the coordinates after spectral initialization to separate duplicate observations (e.g., to avoid overplotting).
     * This is done using normally-distributed noise of mean zero and standard deviation of `Options::initialize_spectral_jitter_sd`.
     * Only relevant if `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization does not fail.
     */
    bool initialize_spectral_jitter = false;

    /**
     * Standard deviation of the jitter to apply after spectral initialization.
     * Only relevant if `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization does not fail and `Options::initialize_spectral_jitter = true`.
     */
    double initialize_spectral_jitter_sd = 0.0001;

    /**
     * Scale of the randomly generated initial coordinates.
     * Specifically, Coordinates are sampled from a uniform distribution from \f$[-x, x)\f$ where \f$x\f$ is `initialize_random_scale`.
     * Only relevant if `Options::initialize_method = InitializeMethod::RANDOM`,
     * or `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization fails and `Options::initialize_random_on_spectral_fail = true`.
     */
    double initialize_random_scale = 10;

    /**
     * Seed for the random number generator during initialization.
     * Only relevant if `Options::initialize_method = InitializeMethod::RANDOM`;
     * or `Options::initialize_method = InitializeMethod::SPECTRAL` and `Options::initialize_spectral_jitter = true`;
     * or `Options::initialize_method = InitializeMethod::SPECTRAL` and spectral initialization fails and `Options::initialize_random_on_spectral_fail = true`.
     */
    typename RngEngine::result_type initialize_seed = sanisizer::cap<typename RngEngine::result_type>(9876543210);

    /**
     * Number of epochs for the gradient descent, i.e., optimization iterations. 
     * Larger values improve accuracy at the cost of increased compute time.
     * If no value is provided, one is automatically chosen based on the size of the dataset:
     *
     * - For datasets with no more than 10000 observations, the number of epochs is set to 500.
     * - For larger datasets with more than 10000 observations, the number of epochs is inversely proportional to the number of observations.
     *   Specifically, the number of epochs starts at 500 for 10000 observations and decreases asymptotically to a lower limit of 200.
     *   This choice aims to reduce computational work for very large datasets. 
     */
    std::optional<int> num_epochs;

    /**
     * Initial learning rate used in the gradient descent.
     * Larger values can accelerate convergence but at the risk of skipping over suitable local optima.
     */
    double learning_rate = 1; 

    /**
     * Rate of sampling negative observations to compute repulsive forces.
     * Greater values will improve accuracy but increase compute time. 
     */
    double negative_sample_rate = 5;

    /**
     * Number of neighbors to use to define the fuzzy sets.
     * Larger values improve connectivity and favor preservation of global structure, at the cost of increased compute time.
     * This argument is only used in certain `initialize()` overloads that perform identification of the nearest neighbors. 
     */
    int num_neighbors = 15;

    /**
     * Seed for the random number generator when sampling negative observations in the optimization step.
     */
    typename RngEngine::result_type optimize_seed = sanisizer::cap<typename RngEngine::result_type>(1234567890);

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
     * Nonetheless, users can enable parallel optimization if cost is no issue - usually a higher number of threads (above 4) is required to see a significant speed-up.
     *
     * If the `UMAPPP_NO_PARALLEL_OPTIMIZATION` macro is defined, **umappp** will not be compiled with support for parallel optimization.
     * This may be desirable in environments that have no support for threading or atomics, or to reduce the binary size if parallelization is not of interest.
     * In such cases, enabling parallel optimization and calling `Status::run()` will throw an error.
     */
    int parallel_optimization = false;
};

}

#endif
