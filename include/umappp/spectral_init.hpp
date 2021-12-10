#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#include "irlba/irlba.hpp"
#include "Eigen/Sparse"

#include <vector>
#include <random>
#include <algorithm>

#include "NeighborList.hpp"
#include "aarand/aarand.hpp"

namespace umappp {

/* Peeled from the function of the same name in the uwot package,
 * see https://github.com/jlmelville/uwot/blob/master/R/init.R for details.
 */
template<typename Float>
bool normalized_laplacian(const NeighborList<Float>& edges, int ndim, Float* Y) {
    std::vector<double> sums(edges.size());
    std::vector<int> sizes(edges.size());

    for (size_t c = 0; c < edges.size(); ++c) {
        const auto& current = edges[c];
        sizes[c] = current.size() + 1; // +1 for self, assuming that no entry of 'current' is equal to 'c'.

        double& sum = sums[c];
        for (const auto& f : current) {
            sum += f.second;
        }
        sum = std::sqrt(sum);
    }

    // Creating a normalized sparse matrix.
    Eigen::SparseMatrix<double> mat(edges.size(), edges.size());
    mat.reserve(sizes);

    for (size_t c = 0; c < edges.size(); ++c) {
        const auto& current = edges[c]; 
        size_t i = 0;

        for (; i < current.size() && current[i].first < c; ++i) {
            const auto& f = current[i];
            mat.insert(f.first, c) = -f.second / sums[f.first] / sums[c];
        }

        mat.insert(c, c) = 1;

        for (; i < current.size(); ++i) {
            const auto& f = current[i];
            mat.insert(f.first, c) = -f.second / sums[f.first] / sums[c];
        }
    }
    mat.makeCompressed();

    /* We want to find the eigenvectors corresponding to the 'ndim' smallest
     * positive eigenvalues, as these define a nice initial partition of the
     * observations (i.e., weak-to-no edges = small eigenvalues). Unfortunately,
     * the best algorithms are designed to find the largest eigenvalues/vectors. 
     * 
     * So, we observe that the normalized laplacian is positive semi-definite
     * where the smallest eigenvalue is zero and the largest _possible_
     * eigenvalue is 2. Thus, we shift the matrix (i.e., '2 * I - L') and then
     * finding the largest 'ndim + 1' eigenvalues from the shifted matrix.
     * These correspond to the smallest 'ndim + 1' eigenvalues from the
     * original matrix. This is obvious when we realize that the eigenvectors
     * of A are the same as the eigenvectors of (xI - A), but the order of
     * eigenvalues is reversed because of the negation.
     *
     * Initially motivated by comments at yixuan/spectra#126 but I misread the
     * equations so this approach (while correct) is not what is described in
     * those links. Also thanks to jlmelville for the max eigenvalue hint,
     * see LTLA/umappp#4 for the discussion.
     */
    mat *= -1;
    mat.diagonal().array() += static_cast<double>(2);

    irlba::Irlba runner;
    auto actual = runner.set_number(ndim + 1).run(mat);
    auto ev = actual.U.rightCols(ndim); 

    // Getting the maximum value; this is assumed to be non-zero,
    // otherwise this entire thing is futile.
    const double max_val = std::max(std::abs(ev.minCoeff()), std::abs(ev.maxCoeff()));
    const double expansion = (max_val > 0 ? 10 / max_val : 1);

    for (size_t c = 0; c < edges.size(); ++c) {
        size_t offset = c * ndim;
        for (int d = 0; d < ndim; ++d) {
            Y[offset + d] = ev(c, d) * expansion; // TODO: put back the jitter step.
        }
    }
    return true;
}

template<typename Float>
bool has_multiple_components(const NeighborList<Float>& edges) {
    if (!edges.size()) {
        return false;
    }

    size_t in_component = 1;
    std::vector<int> remaining(1, 0);
    std::vector<int> mapping(edges.size(), -1);
    mapping[0] = 0;

    do {
        int curfriend = remaining.back();
        remaining.pop_back();

        for (const auto& ff : edges[curfriend]) {
            if (mapping[ff.first] == -1) {
                remaining.push_back(ff.first);
                mapping[ff.first] = 0;
                ++in_component;
            }
        }
    } while (remaining.size());

    return in_component != edges.size();
}

template<typename Float>
bool spectral_init(const NeighborList<Float>& edges, int ndim, Float* vals) {
    if (!has_multiple_components(edges)) {
        if (normalized_laplacian(edges, ndim, vals)) {
            return true;
        }
    }
    return false;
}

template<typename Float>
void random_init(size_t nobs, int ndim, Float * vals) {
    std::mt19937_64 rng(nobs * ndim); // for a bit of deterministic variety.
    for (size_t i = 0; i < nobs * ndim; ++i) {
        vals[i] = aarand::standard_uniform<Float>(rng) * static_cast<Float>(20) - static_cast<Float>(10); // values from (-10, 10).
    }
    return;
}

}

#endif
