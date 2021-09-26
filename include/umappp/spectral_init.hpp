#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#include "Spectra/SymEigsSolver.h"
#include "Spectra/MatOp/SparseSymMatProd.h"
//#include "Spectra/SymEigsShiftSolver.h"
//#include "Spectra/MatOp/SparseSymShiftSolve.h"
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
inline bool normalized_laplacian(const NeighborList& edges, int ndim, double* Y) {
    std::vector<double> sums(edges.size());
    std::vector<int> sizes(edges.size());

    for (size_t c = 0; c < edges.size(); ++c) {
        const auto& current = edges[c];
        int& count = sizes[c];
        double& sum = sums[c];

        for (const auto& f : current) {
            sum += f.second;
            count += (f.first <= c);
        }

        sum = std::sqrt(sum);
        ++count; // for self, assuming that f.first != c.
    }

    // Creating a normalized sparse matrix.
    Eigen::SparseMatrix<double> mat(edges.size(), edges.size());
    mat.reserve(sizes);

    for (size_t c = 0; c < edges.size(); ++c) {
        const auto& current = edges[c]; 
        for (const auto& f : current) {
            if (c < f.first) { // upper-triangular only.
                break;
            }
            mat.insert(f.first, c) = -f.second / sums[f.first] / sums[c];
        }
        mat.insert(c, c) = 1;
    }
    mat.makeCompressed();

    // Finding the smallest eigenvalues & their eigenvectors,
    // using the shift-and-invert mode as recommended.
    const int nobs = mat.rows();
    int nev = std::min(ndim + 1, nobs); // +1 from uwot:::normalized_laplacian_init
    int ncv = std::min(nobs, std::max(2 * nev, 20)); // from RSpectra:::eigs_real_sym. I don't make the rules.

    Spectra::SparseSymMatProd<double, Eigen::Upper> op(mat);
    Spectra::SymEigsSolver<typename std::remove_reference<decltype(op)>::type> eigs(op, nev, ncv); 

    eigs.init();
    eigs.compute(Spectra::SortRule::SmallestMagn);

//    Spectra::SparseSymShiftSolve<double, Eigen::Upper> op(mat);
//    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double, Eigen::Upper> > eigs(op, nev, ncv, -0.001); // see https://github.com/yixuan/spectra/issues/126
//
//    eigs.init();
//    eigs.compute(Spectra::SortRule::LargestMagn);

    if (eigs.info() != Spectra::CompInfo::Successful) {
        return false;
    }

    auto ev = eigs.eigenvectors().rightCols(ndim); 

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

inline bool has_multiple_components(const NeighborList& edges) {
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

inline bool spectral_init(const NeighborList& edges, int ndim, double * vals) {
    if (!has_multiple_components(edges)) {
        if (normalized_laplacian(edges, ndim, vals)) {
            return true;
        }
    }
    return false;
}

inline void random_init(size_t nobs, int ndim, double * vals) {
    std::mt19937_64 rng(nobs * ndim); // for a bit of deterministic variety.
    for (size_t i = 0; i < nobs * ndim; ++i) {
        vals[i] = aarand::standard_uniform(rng) * 20 - 10; // values from (-10, 10).
    }
    return;
}

}

#endif
