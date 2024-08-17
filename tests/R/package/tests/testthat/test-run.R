# library(testthat); library(umappp); source("test-run.R")

# Generating some data.
set.seed(10)
y <- matrix(rnorm(50, sd=2), ncol=10)
id <- sample(nrow(y), 500, replace=TRUE)
mat <- y[id,] + matrix(rnorm(5000), ncol=10)

library(BiocNeighbors)
res <- findKNN(mat, k=15) 

library(uwot)
library(testthat)

test_that("initialization is done correctly", {
    obs <- umappp:::initialize_umap(t(res$index - 1L), t(res$distance), ndim=2)

    # Reference calculation.
    d2sr <- uwot:::data2set(mat, NULL, metric="precomputed", method="umap",
        nn_method = list(idx = cbind(1:nrow(mat), res$index), dist = cbind(0, res$distance)),
        local_connectivity = 1,
        bandwidth = 1, 
        n_threads=1, 
        grain_size=1,
        set_op_mix_ratio=1,
        ret_sigma=FALSE)

    V <- d2sr$V
    init <- uwot:::spectral_init(d2sr$V)
    rescale <- sign(colSums(init)/colSums(obs[[1]]))
    init <- sweep(init, 2, rescale, "*")
    expect_equal(init, obs[[1]], tol=1e-3) # higher tolerance required because uwot jiggles the inputs a bit.

    n_epochs <- 500
    V@x[V@x < max(V@x)/n_epochs] <- 0
    V <- Matrix::drop0(V)
    epochs_per_sample <- uwot:::make_epochs_per_sample(V@x, n_epochs)
    expect_equal(epochs_per_sample, obs[[2]][[3]])
})

set.seed(10000)
test_that("general run is not too inconsistent", {
    ref <- uwot::umap(X = mat, nn_method=list(idx=cbind(1:nrow(mat), res$index), dist=cbind(0, res$distance)), a=2, b=1)
    obs <- runUmap(res$index, res$distance, ndim=2, a=2, b=1, seed=100)

    # Values are within a similar range and scale.
    expect_true(all(obs < 15 & obs > -15))
    expect_true(all(ref < 15 & ref > -15))

    expect_gt(var(ref[,1]), 10)
    expect_gt(var(obs[,1]), 10)
    expect_gt(var(ref[,2]), 10)
    expect_gt(var(obs[,2]), 10)

    png("demo.png", width=10, height=5, units="in", res=120)
    par(mfrow=c(1,2))
    plot(ref[,1], ref[,2], col=id, main="uwot")
    plot(obs[,1], obs[,2], col=id, main="umappp")
    dev.off()
})
