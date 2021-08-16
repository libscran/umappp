#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP

#include "Spectra/SymEigsShiftSolver.h"
#include "Spectra/MatOp/SparseSymShiftSolve.h"
#include "Eigen/Sparse"

#include <vector>
#include <unordered_map>
#include <random>

namespace umappp {

typedef std::vector<Eigen::Triplet<double> > Edges;

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
inline Eigen::MatrixXd normalized_laplacian_init(Edges edges, int order, int ndim) {
    std::vector<double> sums(order);

    // Normalizing the values.
    for (const auto& e : edges) {
        sums[e.row()] += e.value();
        if (e.row() != e.col()) {
            sums[e.col()] += e.value();
        }
    }

    for (auto& s : sums) {
        s = std::sqrt(s);
    }

    for (auto& e : edges) {
        double val = -e.value() / (sums[e.row()] * sums[e.col()]);
        if (e.row() == e.col()) {
            val = 1 - val;
        }
        e = Eigen::Triplet<double>(e.row(), e.col(), val);
    }

    // Finding the smallest eigenvalues & their eigenvectors,
    // using the shift-and-invert mode as recommended.
    Eigen::SparseMatrix<double> mat(order, order);
    mat.setFromTriplets(edges.begin(), edges.end());
    mat.makeCompressed();

    Spectra::SparseSymShiftSolve<double> op(mat);
    Spectra::SymEigsShiftSolver<decltype(op)> eigs(op, 
                                                   ndim + 1, // nev 
                                                   2 * (ndim + 1), // following recommendation to take 2 * nev.
                                                   0.0);

    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn);

    if (eigs.info() == Spectra::CompInfo::Successful) {
        return eigs.eigenvectors().leftCols(ndim);

    } else {
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

/* Finds the connected components of the graph. Pretty sure this
 * is not the most efficient way to do it, but whatever.
 */
inline std::pair<int, std::vector<int> > find_components(const Edges& edges, int order) {
    std::unordered_map<int, std::vector<int> > friends;
    for (const auto& e : edges) {
        friends[e.row()].push_back(e.col());
        friends[e.col()].push_back(e.row());
    }
    
    std::pair<int, std::vector<int> > output(0, std::vector<int>(order, -1));
    int& counter = output.first;
    auto& mapping = output.second;

    for (int current = 0; current< order; ++current) {
        if (mapping[current] != -1) {
            continue;
        }

        std::vector<int> remaining(1, current);
        mapping[current] = counter;
        do {
            int curfriend = remaining.back();
            remaining.pop_back();

            for (auto ff : friends[curfriend]) {
                if (mapping[ff] == -1) {
                    remaining.push_back(ff);
                    mapping[ff] = counter;
                }
            }
        } while (remaining.size());

        ++counter;
    }

    return output;
}

inline void spectral_init(const Edges& edges, int order, int ndim, double * vals) {
    auto mapping = find_components(edges, order);

    // Building the new indices.
    std::vector<int> cumulative(mapping.first);
    std::vector<int> new_indices(order);
    std::vector<std::vector<int> > reversed(mapping.first);

    for (size_t i = 0; i < mapping.second.size(); ++i) {
        auto m = mapping.second[i];
        new_indices[i] = cumulative[m];
        reversed[m].push_back(i);
        ++cumulative[m];
    }

    // Splitting nodes into their categories.
    std::vector<Edges> components(mapping.first);
    for (const auto& e : edges) {
        int comp = mapping.second[e.row()];
        components[comp].emplace_back(new_indices[e.row()], new_indices[e.col()], e.value());
    }

    // Initializing each of them separately.
    for (size_t c = 0; c < components.size(); ++c) {
        auto eigs = normalized_laplacian_init(std::move(components[c]), reversed[c].size(), ndim);

        for (size_t r = 0; r < reversed[c].size(); ++r) {
            size_t offset = reversed[c][r] * ndim;
            for (int d = 0; d < ndim; ++d) {
                vals[offset + d] = eigs(r, d);
            }
        }
    }
}

}

#endif
