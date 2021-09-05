#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#include "Spectra/SymEigsShiftSolver.h"
#include "Spectra/MatOp/SparseSymShiftSolve.h"
#include "Eigen/Sparse"

#include <vector>
#include <random>
#include <algorithm>

#include "find_components.hpp"
#include "NeighborList.hpp"

namespace umappp {

/* Peeled from the function of the same name in the uwot package,
 * see https://github.com/jlmelville/uwot/blob/master/R/init.R for details.
 *
 * Edges are assumed to be such that the larger node index is first,
 * i.e., the corresponding adjacency matrix is lower triangular. This 
 * fits with the defaults for SparseSymShiftSolve.
 *
 * We also assume that there is always a self-edge, i.e., the diagonal
 * of the adjacency matrix is non-zero. 
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
        auto current = edges[which[c]]; // deliberate copy.
        std::sort(current.begin(), current.end());

        for (const auto& f : current) {
            auto new_index = indices[f.first];
            if (c < new_index) { // lower-triangular only.
                break;
            }

            double val = f.second / sums[new_index] / sums[c];
            if (c == new_index) {
                val = 1 - val;
            } 
            mat.insert(new_index, c) = val;
        }
    }
    mat.makeCompressed();

    // Finding the smallest eigenvalues & their eigenvectors,
    // using the shift-and-invert mode as recommended.
    // We follow Spectra's recommendation to take 2 * nev.
    int nev = std::min(ndim + 1, static_cast<int>(mat.rows()));
    Spectra::SparseSymShiftSolve<double> op(mat);
    Spectra::SymEigsShiftSolver<decltype(op)> eigs(op, nev, 2 * nev, 0.0); 

    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn);

    if (eigs.info() == Spectra::CompInfo::Successful) {
        return eigs.eigenvectors().leftCols(ndim);

    } else {
        size_t order = which.size();

        // Falling back to random initialization.
        Eigen::MatrixXd output(order, ndim);
        std::mt19937_64 rng(1234567890);
        std::uniform_real_distribution<> dist(0, 1);

        auto outptr = output.data();
        for (size_t i = 0; i < order * ndim; ++i) {
            outptr[i] = dist(rng);
        }
        return output;
    }
}

inline void spectral_init(const NeighborList& edges, int ndim, double * vals) {
    auto mapping = find_components(edges);

    for (size_t c = 0; c < mapping.ncomponents(); ++c) {
        auto eigs = normalized_laplacian_by_component(edges, mapping, c, ndim);
        const auto& reversed = mapping.reversed[c];

        for (size_t r = 0; r < reversed.size(); ++r) {
            size_t offset = reversed[r] * ndim;
            for (int d = 0; d < ndim; ++d) {
                vals[offset + d] = eigs(r, d);
            }
        }
    }
}

}

#endif
