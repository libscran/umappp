#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#include "Spectra/SymEigsSolver.h"
//#include "Spectra/SymEigsShiftSolver.h"
#include "Spectra/MatOp/SparseSymMatProd.h"
#include "Eigen/Sparse"

#include <vector>
#include <random>
#include <algorithm>

#include "find_components.hpp"
#include "NeighborList.hpp"
#include "aarand/aarand.hpp"

namespace umappp {

/* Peeled from the function of the same name in the uwot package,
 * see https://github.com/jlmelville/uwot/blob/master/R/init.R for details.
 */
inline Eigen::MatrixXd normalized_laplacian_by_component(const NeighborList& edges, const ComponentIndices& comp_info, int component, int ndim) {
    const auto& which = comp_info.reversed[component];
    const auto& indices = comp_info.new_indices;

    std::vector<double> sums(which.size());
    for (size_t c = 0; c < which.size(); ++c) {
        const auto& current = edges[which[c]];
        double& sum = sums[c];
        for (const auto& f : current) {
            sum += f.second;
        }
        sum = std::sqrt(sum);
    }

    // Creating a normalized sparse matrix.
    Eigen::SparseMatrix<double> mat(which.size(), which.size());
    {
        std::vector<int> sizes(which.size());
        for (size_t c = 0; c < which.size(); ++c) {
            const auto& current = edges[which[c]];
            sizes[c] = current.size();
        }
        mat.reserve(sizes);
    }

    for (size_t c = 0; c < which.size(); ++c) {
        const auto& current = edges[which[c]]; 
        for (const auto& f : current) {
            auto new_index = indices[f.first];
            if (c < new_index) { // upper-triangular only.
                break;
            }
            mat.insert(new_index, c) = -f.second / sums[new_index] / sums[c];
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
//    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double, Eigen::Upper> > eigs(op, nev, ncv, 0.0); 
//
//    eigs.init();
//    eigs.compute(Spectra::SortRule::LargestMagn);

    if (eigs.info() == Spectra::CompInfo::Successful) {
//        return eigs.eigenvectors().leftCols(ndim);
        return eigs.eigenvectors().rightCols(ndim);

    } else {
        size_t order = which.size();

        // Falling back to random initialization.
        Eigen::MatrixXd output(order, ndim);
        std::mt19937_64 rng(1234567890);

        auto outptr = output.data();
        for (size_t i = 0; i < order * ndim; ++i) {
            outptr[i] = aarand::standard_uniform(rng) * 20 - 10; // values from (-10, 10).
        }
        return output;
    }
}

inline void spectral_init(const NeighborList& edges, int ndim, double * vals) {
    auto mapping = find_components(edges);

    for (size_t c = 0; c < mapping.ncomponents(); ++c) {
        auto eigs = normalized_laplacian_by_component(edges, mapping, c, ndim);
        const auto& reversed = mapping.reversed[c];

        // Getting the maximum value; this is assumed to be non-zero,
        // otherwise this entire thing is futile.
        const double max_val = std::max(std::abs(eigs.minCoeff()), std::abs(eigs.maxCoeff()));
        const double expansion = (max_val > 0 ? 10 / max_val : 1);

        for (size_t r = 0; r < reversed.size(); ++r) {
            size_t offset = reversed[r] * ndim;
            for (int d = 0; d < ndim; ++d) {
                vals[offset + d] = eigs(r, d) * expansion; // TODO: put back the jitter step.
            }
        }
    }
}

}

#endif
