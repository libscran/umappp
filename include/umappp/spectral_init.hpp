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
    std::vector<Float> sums(edges.size());
    std::vector<int> sizes(edges.size());

    for (size_t c = 0; c < edges.size(); ++c) {
        const auto& current = edges[c];
        sizes[c] = current.size() + 1; // +1 for self, assuming that no entry of 'current' is equal to 'c'.

        Float& sum = sums[c];
        for (const auto& f : current) {
            sum += f.second;
        }
        sum = std::sqrt(sum);
    }

    // Creating a normalized sparse matrix.
    Eigen::SparseMatrix<Float> mat(edges.size(), edges.size());
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

    // Finding the largest eigenvalue, shifting the matrix, and then finding the 
    // largest 'ndim + 1' eigenvalues from that.
    irlba::Irlba runner;
    auto deets = runner.set_number(1).run(mat);
    Float max_eigval = deets.D[0];
    
    mat *= -1;
    mat.diagonal().array() += max_eigval;

    auto actual = runner.set_number(ndim + 1).run(mat);
    auto ev = actual.U.rightCols(ndim); 

    // Getting the maximum value; this is assumed to be non-zero,
    // otherwise this entire thing is futile.
    const Float max_val = std::max(std::abs(ev.minCoeff()), std::abs(ev.maxCoeff()));
    const Float expansion = (max_val > 0 ? 10 / max_val : 1);

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
