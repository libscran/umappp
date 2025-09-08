#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#include <vector>
#include <algorithm>
#include <cstddef>

#include "aarand/aarand.hpp"
#include "irlba/irlba.hpp"
#include "irlba/parallel.hpp"
#include "sanisizer/sanisizer.hpp"

#include "NeighborList.hpp"
#include "Options.hpp"

namespace umappp {

namespace internal {

/* Peeled from the function of the same name in the uwot package,
 * see https://github.com/jlmelville/uwot/blob/master/R/init.R for details.
 *
 * It is assumed that 'edges' has already been symmetrized.
 */
template<typename Index_, typename Float_>
bool normalized_laplacian(const NeighborList<Index_, Float_>& edges, const std::size_t num_dim, Float_* const Y, const irlba::Options& irlba_opt, const int nthreads, double scale) {
    const Index_ nobs = edges.size();
    auto sums = sanisizer::create<std::vector<double> >(nobs); // we deliberately use double-precision to avoid difficult problems from overflow/underflow inside IRLBA.
    std::vector<std::size_t> pointers(sanisizer::sum<typename std::vector<std::size_t>::size_type>(nobs, 1));
    std::size_t reservable = 0;

    for (Index_ c = 0; c < nobs; ++c) {
        const auto& current = edges[c];

        reservable = sanisizer::sum<std::size_t>(reservable, current.size()); 
        reservable = sanisizer::sum<std::size_t>(reservable, 1); // +1 for self, assuming that no entry of 'current' is equal to 'c'.
        pointers[c + 1] = reservable;

        double& sum = sums[c];
        for (const auto& f : current) {
            sum += f.second;
        }
        sum = std::sqrt(sum);
    }

    // Creating a normalized sparse matrix. Everything before TRANSFORM is the
    // actual normalized laplacian, everything after TRANSFORM is what we did
    // to the laplacian to make it possible to get the smallest eigenvectors. 
    std::vector<double> values;
    values.reserve(reservable);
    std::vector<Index_> indices;
    indices.reserve(reservable);

    for (Index_ c = 0; c < nobs; ++c) {
        const auto& current = edges[c]; 
        auto cIt = current.begin();
        const auto cLast = current.end();

        for (; cIt != cLast && cIt->first < c; ++cIt) {
            indices.push_back(cIt->first);
            values.push_back(- static_cast<double>(cIt->second) / sums[cIt->first] / sums[c] /* TRANSFORM */ * (-1) );
        }

        // Adding unity at the diagonal.
        indices.push_back(c); 
        values.push_back(1 /* TRANSFORM */ * (-1) + 2);

        for (; cIt != cLast; ++cIt) {
            indices.push_back(cIt->first);
            values.push_back(- static_cast<double>(cIt->second) / sums[cIt->first] / sums[c] /* TRANSFORM */ * (-1) );
        }
    }

    /* Okay, here's the explanation for the TRANSFORM transformations.
     *
     * We want to find the eigenvectors corresponding to the 'ndim' smallest
     * positive eigenvalues, as these define a nice initial partition of the
     * observations (i.e., weak-to-no edges = small eigenvalues). Unfortunately,
     * the best algorithms are designed to find the largest eigenvalues/vectors. 
     * 
     * So, we observe that the normalized laplacian is positive semi-definite
     * where the smallest eigenvalue is zero and the largest _possible_
     * eigenvalue is 2. Thus, we shift the matrix (i.e., '2 * I - L') and then
     * find the largest 'ndim + 1' eigenvalues from the shifted matrix.  These
     * correspond to the smallest 'ndim + 1' eigenvalues from the original
     * matrix. This is obvious when we realize that the eigenvectors of A are
     * the same as the eigenvectors of (xI - A), but the order of eigenvalues
     * is reversed because of the negation. Then, out of the 'ndim + 1' largest
     * eigenvalues, we remove the largest one, because that corresponds to the
     * smallest eigenvalue of zero in the original matrix - leaving us with
     * eigenvectors for the 'ndim' smallest positive eigenvalues.
     *
     * Initially motivated by comments at yixuan/spectra#126 but I misread the
     * equations so this approach (while correct) is not what is described in
     * those links. Also thanks to jlmelville for the max eigenvalue hint,
     * see LTLA/umappp#4 for the discussion.
     */

    const irlba::ParallelSparseMatrix<
        decltype(values),
        decltype(indices),
        decltype(pointers)
        // Eigen::VectorXd // TODO: deliberately double-precision here, but not available in 2.0.0
    > mat(nobs, nobs, std::move(values), std::move(indices), std::move(pointers), /* column_major = */ true, nthreads);
    irlba::EigenThreadScope tscope(nthreads);

    const auto actual = irlba::compute(mat, num_dim + 1, irlba_opt);
    if (!actual.converged) {
        return false;
    }
    const auto ev = actual.U.rightCols(num_dim); 

    // Getting the maximum value; this is assumed to be non-zero,
    // otherwise this entire thing is futile.
    const double max_val = std::max(std::abs(ev.minCoeff()), std::abs(ev.maxCoeff()));
    const double expansion = (max_val > 0 ? scale / max_val : 1);

    for (Index_ c = 0; c < nobs; ++c) {
        for (std::size_t d = 0; d < num_dim; ++d) {
            Y[sanisizer::nd_offset<std::size_t>(d, num_dim, c)] = ev.coeff(c, d) * expansion;
        }
    }

    return true;
}

template<typename Index_, typename Float_>
bool has_multiple_components(const NeighborList<Index_, Float_>& edges) {
    if (!edges.size()) {
        return false;
    }

    // We assume that 'edges' is symmetric so we can use a simple recursive algorithm.
    decltype(edges.size()) in_component = 1;
    std::vector<Index_> remaining(1, 0);
    std::vector<unsigned char> traversed(edges.size(), 0);
    traversed[0] = 1;

    do {
        int curfriend = remaining.back();
        remaining.pop_back();

        for (const auto& ff : edges[curfriend]) {
            if (traversed[ff.first] == 0) {
                remaining.push_back(ff.first);
                traversed[ff.first] = 1;
                ++in_component;
            }
        }
    } while (remaining.size());

    return in_component != edges.size();
}

template<typename Index_, typename Float_>
bool spectral_init(
    const NeighborList<Index_, Float_>& edges,
    const std::size_t num_dim,
    Float_* const vals,
    const irlba::Options& irlba_opt,
    const int nthreads,
    const double scale,
    const bool jitter,
    const double jitter_sd,
    const RngEngine::result_type seed)
{
    if (has_multiple_components(edges)) {
        return false;
    }

    if (!normalized_laplacian(edges, num_dim, vals, irlba_opt, nthreads, scale)) {
        return false;
    }

    if (jitter) {
        RngEngine rng(seed);
        const auto ntotal = sanisizer::product_unsafe<std::size_t>(num_dim, edges.size());
        const auto half_ntotal = ntotal / 2;
        for (std::size_t i = 0; i < half_ntotal; ++i) {
            const auto sampled = aarand::standard_normal(rng);
            vals[2 * i] += sampled.first * jitter_sd;
            vals[2 * i + 1] += sampled.second * jitter_sd;
        }

        if (ntotal % 2 == 1) {
            const auto sampled = aarand::standard_normal(rng);
            vals[ntotal - 1] += sampled.first * jitter_sd;
        }
    }

    return true;
}

template<typename Index_, typename Float_>
void random_init(
    const Index_ num_obs,
    const std::size_t num_dim,
    Float_ * const vals,
    const RngEngine::result_type seed,
    const double scale)
{
    RngEngine rng(seed);
    const Float_ mult = scale * 2;
    const Float_ shift = scale;
    const auto ntotal = sanisizer::product_unsafe<std::size_t>(num_dim, num_obs);
    for (std::size_t i = 0; i < ntotal; ++i) {
        vals[i] = aarand::standard_uniform<Float_>(rng) * mult - shift; ; // values from (-scale, scale).
    }
    return;
}

}

}

#endif
