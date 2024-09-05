#include "Rcpp.h"

#include <algorithm>

#define UMAPPP_R_PACKAGE_TESTING
#include "umappp/umappp.hpp"

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
            Rcpp::IntegerVector(edata.head.begin(), edata.head.end()),
            Rcpp::IntegerVector(edata.tail.begin(), edata.tail.end()),
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
    opt.seed = seed;
    opt.num_threads = num_threads;
    opt.parallel_optimization = parallel_optimization;

    auto status = umappp::initialize(std::move(x), ndim, static_cast<double*>(output.begin()), opt);
    status.run();

    return Rcpp::transpose(output);
}
