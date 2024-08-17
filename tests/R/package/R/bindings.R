#' @export
runUmap <- function(index, distance, ndim=2, a=1, b=1, seed=42, num.threads=1, parallel.optimization=FALSE) {
    run_umap(
        t(index - 1L),
        t(distance),
        ndim=ndim,
        a=a,
        b=b,
        seed=seed,
        num_threads=num.threads,
        parallel_optimization=parallel.optimization
    )
}
