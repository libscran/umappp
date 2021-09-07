# A C++ library for UMAP

![Unit tests](https://github.com/LTLA/umappp/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/umappp/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/umappp/branch/master/graph/badge.svg?token=IKFEAP5J55)](https://codecov.io/gh/LTLA/umappp)

## Overview 

**umappp** is a header-only C++ implementation of the Uniform Manifold Approximation and Projection (UMAP) algorithm (McInnes, Healy and Melville, 2018).
UMAP is a non-linear dimensionality reduction technique that is most commonly used for visualization of complex datasets. 
This is achieved by placing each observation on a low-dimensional (usually 2D) embedding in a manner that preserves the neighborhood of each observation from the high-dimensional original space.
The aim is to ensure that the local structure of the data is faithfully recapitulated in lower dimensions 
Further theoretical details can be found in the [original UMAP documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html);
the implementation here is derived from the C++ code in the [**uwot** R package](https://github.com/jlmelville/uwot).

## Quick start

Given a pointer to a column-major input array with `ndim` rows and `nobs` columns, we can compute the UMAP embedding easily:

```cpp
#include "umappp/Umap.hpp"

std::vector<double> embedding(npts * 2);
umappp::Umap x;
x.run(ndim, nobs, data.data(), 2, embedding.data());
```

We can modify parameters by calling the relevant setters in the `Umap` class:

```cpp
umappp::Umap x;
x.set_num_neighbors(20).set_num_epochs(200);
```

We can initialize and run the algorithm up to the specified number of epochs:

```cpp
umappp::Umap x;

auto status = x.initialize(ndim, nobs, data.data(), 2, embedding.data());

for (int iter = 10; iter < 200; iter += 10) {
    x.run(status, 2, embedding.data(), iter);
    // do something with the current embedding, e.g., create an animation
}
```

Advanced users can control the neighbor search by either providing the search results directly (as a vector of vectors of index-distance pairs)
or by providing an appropriate [**knncolle**](https://github.com/LTLA/knncolle) subclass to the `run()` or `initialize()` functions:

```cpp
umappp::Umap x;
knncolle::AnnoyEuclidean<> searcher(ndim, nobs, data.data());
x.run(&searcher, 2, embedding.data());
```

See the [reference documentation](https://ltla.github.io/umappp) for more details.

## Building projects

If you're already using CMake, you can add something like this to your `CMakeLists.txt`:

```
include(FetchContent)

FetchContent_Declare(
  umappp 
  GIT_REPOSITORY https://github.com/LTLA/umappp
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(umappp)
```

And then:

```
# For executables:
target_link_libraries(myexe umappp)

# For libaries
target_link_libraries(mylib INTERFACE umappp)
```

Otherwise, you can just copy the directory in `include` into some location that is visible to your compiler.
Note that this requires the additional dependencies listed in `extern`:

- The [**Spectra**](https://github.com/yixuan/spectra) library for eigenvalue computations.
- The [**knncolle**](https://github.com/LTLA/knncolle) library for nearest neighbor search.
If you are instead supplying your own neighbor search, this dependency can be eliminated by defining the `UMAPPP_CUSTOM_NEIGHBORS` macro.

## References

McInnes L, Healy J, Melville J (2020).
UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
_arXiv_, https://arxiv.org/abs/1802.03426
