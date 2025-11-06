#include "Rcpp.h"

#include <algorithm>

#define UMAPPP_R_PACKAGE_TESTING
#include "umappp/umappp.hpp"

//[[Rcpp::export(rng=false)]]
Rcpp::List create_probabilities(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances) {
    int nr = indices.nrow(), nc = indices.ncol();
    umappp::NeighborList<int, double> x(nc);
    for (int i = 0; i < nc; ++i) {
        auto curi = indices.column(i);
        auto curd = distances.column(i);
        for (int j = 0; j < nr; ++j) { 
            x[i].emplace_back(curi[j], curd[j]);
        }
    }

    umappp::NeighborSimilaritiesOptions<double> nsopt;
    nsopt.local_connectivity = 1;
    nsopt.bandwidth = 1;
    nsopt.num_threads = 1;
    umappp::neighbor_similarities(x, nsopt);

    umappp::combine_neighbor_sets(x, 1.0);

    std::size_t total_size = 0;
    Rcpp::IntegerVector pointers(nc + 1);
    for (int c = 0; c < nc; ++c) {
        total_size += x[c].size();
        pointers[c + 1] = total_size;
    }

    Rcpp::IntegerVector new_indices(total_size);
    Rcpp::NumericVector new_probs(total_size);
    total_size = 0;
    for (int c = 0; c < nc; ++c) {
        const auto& current = x[c];
        for (const auto& p : current) {
            new_indices[total_size] = p.first;
            new_probs[total_size] = p.second;
            ++total_size;
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("x") = new_probs,
        Rcpp::Named("i") = new_indices,
        Rcpp::Named("p") = pointers
    );
}

//[[Rcpp::export(rng=false)]]
Rcpp::List initialize_umap(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances, int ndim) {
    int nr = indices.nrow(), nc = indices.ncol();
    umappp::NeighborList<int, double> x(nc);
    for (int i = 0; i < nc; ++i) {
        auto curi = indices.column(i);
        auto curd = distances.column(i);
        for (int j = 0; j < nr; ++j) { 
            x[i].emplace_back(curi[j], curd[j]);
        }
    }

    Rcpp::NumericMatrix output(ndim, nc);
    auto status = umappp::initialize(std::move(x), ndim, static_cast<double*>(output.begin()), umappp::Options());

    const auto& edata = status.get_epoch_data();
    return Rcpp::List::create(
        Rcpp::transpose(output),
        Rcpp::List::create(
            Rcpp::IntegerVector(edata.cumulative_num_edges.begin() + 1, edata.cumulative_num_edges.end()),
            Rcpp::IntegerVector(edata.edge_targets.begin(), edata.edge_targets.end()),
            Rcpp::NumericVector(edata.epochs_per_sample.begin(), edata.epochs_per_sample.end())
        )
    );
}

//[[Rcpp::export(rng=false)]]
Rcpp::NumericMatrix run_umap(
    Rcpp::IntegerMatrix indices,
    Rcpp::NumericMatrix distances,
    int ndim,
    double a,
    double b,
    int seed,
    int num_threads,
    bool parallel_optimization) 
{
    int nr = indices.nrow(), nc = indices.ncol();
    umappp::NeighborList<int, double> x(nc);
    for (int i = 0; i < nc; ++i) {
        auto curi = indices.column(i);
        auto curd = distances.column(i);
        for (int j = 0; j < nr; ++j) { 
            x[i].emplace_back(curi[j], curd[j]);
        }
    }

    Rcpp::NumericMatrix output(ndim, nc);
    umappp::Options opt;
    opt.a = a;
    opt.b = b;
    opt.optimize_seed = seed;
    opt.num_threads = num_threads;
    opt.parallel_optimization = parallel_optimization;

    auto optr = static_cast<double*>(output.begin());
    auto status = umappp::initialize(std::move(x), ndim, optr, opt);
    status.run(optr);

    return Rcpp::transpose(output);
}
