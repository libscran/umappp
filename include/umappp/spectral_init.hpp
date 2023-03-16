#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#ifndef IRLBA_CUSTOM_PARALLEL
#ifdef UMAPPP_CUSTOM_PARALLEL
namespace umappp {

template<class Function>
void irlba_parallelize_(int nthreads, Function fun) {
    UMAPPP_CUSTOM_PARALLEL(nthreads, [&](size_t f, size_t l) -> void {
        // This loop should be trivial if f + 1== l when nthreads == njobs.
        // Nonetheless, we still have a loop just in case the arbitrary
        // scheduling does wacky things. 
        for (size_t i = f; i < l; ++i) {
            fun(f);
        }
    }, nthreads);
}

}
#define IRLBA_CUSTOM_PARALLEL umappp::irlba_parallelize_
#endif
#endif

#include "irlba/irlba.hpp"
#include "irlba/parallel.hpp"

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
bool normalized_laplacian(const NeighborList<Float>& edges, int ndim, Float* Y, int nthreads) {
    size_t nobs = edges.size();
    std::vector<double> sums(nobs);
    std::vector<size_t> pointers;
    pointers.reserve(nobs + 1);
    pointers.push_back(0);
    size_t reservable = 0;

    for (size_t c = 0; c < nobs; ++c) {
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
    std::vector<int> indices;
    indices.reserve(reservable);

    for (size_t c = 0; c < nobs; ++c) {
        const auto& current = edges[c]; 
        auto cIt = current.begin(), last = current.end();

        for (; cIt != last && cIt->first < c; ++cIt) {
            indices.push_back(cIt->first);
            values.push_back(- cIt->second / sums[cIt->first] / sums[c] /* TRANSFORM */ * (-1) );
        }

        // Adding unity at the diagonal.
        indices.push_back(c); 
        values.push_back(1 /* TRANSFORM */ * (-1) + 2);

        for (; cIt != current.end(); ++cIt) {
            indices.push_back(cIt->first);
            values.push_back(- cIt->second / sums[cIt->first] / sums[c] /* TRANSFORM */ * (-1) );
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

    irlba::ParallelSparseMatrix<> mat(nobs, nobs, std::move(values), std::move(indices), std::move(pointers), nthreads);
    irlba::EigenThreadScope tscope(nthreads);

    irlba::Irlba runner;
    auto actual = runner.set_number(ndim + 1).run(mat);
    auto ev = actual.U.rightCols(ndim); 

    // Getting the maximum value; this is assumed to be non-zero,
    // otherwise this entire thing is futile.
    const double max_val = std::max(std::abs(ev.minCoeff()), std::abs(ev.maxCoeff()));
    const double expansion = (max_val > 0 ? 10 / max_val : 1);

    for (size_t c = 0; c < nobs; ++c) {
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
bool spectral_init(const NeighborList<Float>& edges, int ndim, Float* vals, int nthreads) {
    if (!has_multiple_components(edges)) {
        if (normalized_laplacian(edges, ndim, vals, nthreads)) {
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
