# Building the function.
library(Rcpp)
if (!file.exists("umappp")) {
    file.symlink("../../include/umappp", "umappp")
}
sourceCpp("test.cpp")

# Generating some data.
set.seed(10)
y <- matrix(rnorm(50, sd=2), ncol=10)
mat <- y[sample(nrow(y), 500, replace=TRUE),] + matrix(rnorm(5000), ncol=10)

library(FNN)
res <- FNN::get.knn(mat, k=15) # this needs to be the default perplexity * 3.

library(uwot)
ref <- uwot::umap(X = mat, nn_method=list(idx=cbind(1:nrow(mat), res$nn.index), dist=cbind(0, res$nn.dist)), a=2, b=1)
obs <- run_umap(t(res$nn.index - 1L), t(res$nn.dist), 2, 2, 1)

par(mfrow=c(1,2))
plot(ref[,1], ref[,2])
plot(obs[,1], obs[,2])
