#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#include "irlba/irlba.hpp"
#include "irlba/parallel.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <cstddef>

#include "NeighborList.hpp"
#include "aarand/aarand.hpp"

namespace umappp {

namespace internal {

/* Peeled from the function of the same name in the uwot package,
 * see https://github.com/jlmelville/uwot/blob/master/R/init.R for details.
 *
 * It is assumed that 'edges' has already been symmetrized.
 */
template<typename Index_, typename Float_>
bool normalized_laplacian(const NeighborList<Index_, Float_>& edges, std::size_t num_dim, Float_* Y, int nthreads) {
    Index_ nobs = edges.size();
    std::vector<double> sums(nobs); // we deliberately use double-precision to avoid difficult problems from overflow/underflow inside IRLBA.
    std::vector<std::size_t> pointers;
    pointers.reserve(nobs + 1);
    pointers.push_back(0);
    std::size_t reservable = 0;

    for (Index_ c = 0; c < nobs; ++c) {
        const auto& current = edges[c];

        // +1 for self, assuming that no entry of 'current' is equal to 'c'.
        reservable += current.size() + 1; 
        pointers.push_back(reservable);

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
        auto cIt = current.begin(), last = current.end();

        for (; cIt != last && cIt->first < c; ++cIt) {
            indices.push_back(cIt->first);
            values.push_back(- static_cast<double>(cIt->second) / sums[cIt->first] / sums[c] /* TRANSFORM */ * (-1) );
        }

        // Adding unity at the diagonal.
        indices.push_back(c); 
        values.push_back(1 /* TRANSFORM */ * (-1) + 2);

        for (; cIt != current.end(); ++cIt) {
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

    irlba::ParallelSparseMatrix<
        decltype(values),
        decltype(indices),
        decltype(pointers)
        // Eigen::VectorXd // TODO: deliberately double-precision here, but not available in 2.0.0
    > mat(nobs, nobs, std::move(values), std::move(indices), std::move(pointers), /* column_major = */ true, nthreads);
    irlba::EigenThreadScope tscope(nthreads);

    irlba::Options opt;
    auto actual = irlba::compute(mat, num_dim + 1, opt);
    auto ev = actual.U.rightCols(num_dim); 

    // Getting the maximum value; this is assumed to be non-zero,
    // otherwise this entire thing is futile.
    const double max_val = std::max(std::abs(ev.minCoeff()), std::abs(ev.maxCoeff()));
    const double expansion = (max_val > 0 ? 10 / max_val : 1);

    for (Index_ c = 0; c < nobs; ++c) {
        for (std::size_t d = 0; d < num_dim; ++d, ++Y) {
            *Y = ev.coeff(c, d) * expansion; // TODO: put back the jitter step.
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
bool spectral_init(const NeighborList<Index_, Float_>& edges, std::size_t num_dim, Float_* vals, int nthreads) {
    if (!has_multiple_components(edges)) {
        if (normalized_laplacian(edges, num_dim, vals, nthreads)) {
            return true;
        }
    }
    return false;
}

template<typename Index_, typename Float_>
void random_init(Index_ num_obs, std::size_t num_dim, Float_ * vals) {
    std::size_t num = static_cast<std::size_t>(num_obs) * num_dim; // cast to avoid overflow.
    std::mt19937_64 rng(num); // for a bit of deterministic variety.
    for (std::size_t i = 0; i < num; ++i) {
        vals[i] = aarand::standard_uniform<Float_>(rng) * static_cast<Float_>(20) - static_cast<Float_>(10); // values from (-10, 10).
    }
    return;
}

}

}

#endif
