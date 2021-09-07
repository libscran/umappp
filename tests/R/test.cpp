#include "Rcpp.h"
#include <algorithm>
#include <iostream>

#define UMAPPP_CUSTOM_NEIGHBORS
#include "umappp/Umap.hpp"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::NumericMatrix run_umap(Rcpp::IntegerMatrix indices, Rcpp::NumericMatrix distances, int ndim, double a, double b) {
    int nr = indices.nrow(), nc = indices.ncol();
    umappp::NeighborList x(nc);
    for (int i = 0; i < nc; ++i) {
        auto curi = indices.column(i);
        auto curd = distances.column(i);
        for (int j = 0; j < nr; ++j) { 
            x[i].emplace_back(curi[j], curd[j]);
        }
    }

    Rcpp::NumericMatrix output(ndim, nc);
    umappp::Umap runner;
    runner.set_a(a).set_b(b);
    auto status = runner.run(std::move(x), ndim, (double*)output.begin());

    return Rcpp::transpose(output);
}
