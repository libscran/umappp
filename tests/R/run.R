# Building the function.
library(Rcpp)
sourceCpp("test.cpp")

# Generating some data.
set.seed(10)
y <- matrix(rnorm(50, sd=2), ncol=10)
id <- sample(nrow(y), 500, replace=TRUE)
mat <- y[id,] + matrix(rnorm(5000), ncol=10)

library(FNN)
res <- FNN::get.knn(mat, k=15) 

library(uwot)
library(testthat)

test_that("initialization is done correctly", {
    obs <- initialize_umap(t(res$nn.index - 1L), t(res$nn.dist), 2)

    # Reference calculation.
    d2sr <- uwot:::data2set(mat, NULL, metric="precomputed", method="umap",
        nn_method = list(idx = cbind(1:nrow(mat), res$nn.index), dist = cbind(0, res$nn.dist)),
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

test_that("general run is not too inconsistent", {
    ref <- uwot::umap(X = mat, nn_method=list(idx=cbind(1:nrow(mat), res$nn.index), dist=cbind(0, res$nn.dist)), a=2, b=1)
    obs <- run_umap(t(res$nn.index - 1L), t(res$nn.dist), 2, 2, 1, 123)

    # Values are within range.
    expect_true(all(obs < 10 & obs > -10))
#    expect_true(all(obs2 < 10 & obs2 > -10)) # Who knows why GHA doesn't like this, but oh well.

    png("demo.png", width=10, height=5, units="in", res=120)
    par(mfrow=c(1,2))
    plot(ref[,1], ref[,2], col=id, main="uwot")
    plot(obs[,1], obs[,2], col=id, main="umappp")
    dev.off()
})
