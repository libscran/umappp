#include "Rcpp.h"
#include <algorithm>
#include <iostream>

#define UMAPPP_CUSTOM_NEIGHBORS
#include "umappp/Umap.hpp"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::List initialize_umap(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances, int ndim) {
    int nr = indices.nrow(), nc = indices.ncol();
    umappp::NeighborList<> x(nc);
    for (int i = 0; i < nc; ++i) {
        auto curi = indices.column(i);
        auto curd = distances.column(i);
        for (int j = 0; j < nr; ++j) { 
            x[i].emplace_back(curi[j], curd[j]);
        }
    }

    Rcpp::NumericMatrix output(ndim, nc);
    umappp::Umap<> runner;
    auto status = runner.initialize(std::move(x), ndim, (double*)output.begin());

    return Rcpp::List::create(
        Rcpp::transpose(output),
        Rcpp::List::create(
            Rcpp::IntegerVector(status.epochs.head.begin(), status.epochs.head.end()),
            Rcpp::IntegerVector(status.epochs.tail.begin(), status.epochs.tail.end()),
            Rcpp::NumericVector(status.epochs.epochs_per_sample.begin(), status.epochs.epochs_per_sample.end())
        )
    );
}

// [[Rcpp::export(rng=false)]]
Rcpp::NumericMatrix run_umap(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances, int ndim, double a, double b, int seed) {
    int nr = indices.nrow(), nc = indices.ncol();
    umappp::NeighborList<> x(nc);
    for (int i = 0; i < nc; ++i) {
        auto curi = indices.column(i);
        auto curd = distances.column(i);
        for (int j = 0; j < nr; ++j) { 
            x[i].emplace_back(curi[j], curd[j]);
        }
    }

    Rcpp::NumericMatrix output(ndim, nc);
    umappp::Umap<> runner;
    runner.set_a(a).set_b(b).set_seed(seed);
    auto status = runner.run(std::move(x), ndim, (double*)output.begin());

    return Rcpp::transpose(output);
}
