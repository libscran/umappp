# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

initialize_umap <- function(indices, distances, ndim) {
    .Call('_umappp_initialize_umap', PACKAGE = 'umappp', indices, distances, ndim)
}

run_umap <- function(indices, distances, ndim, a, b, seed, num_threads, parallel_optimization) {
    .Call('_umappp_run_umap', PACKAGE = 'umappp', indices, distances, ndim, a, b, seed, num_threads, parallel_optimization)
}

