# A C++ library for UMAP

![Unit tests](https://github.com/libscran/umappp/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/libscran/umappp/actions/workflows/doxygenate.yaml/badge.svg)
![uwot comparison](https://github.com/libscran/umappp/actions/workflows/compare-uwot.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/libscran/umappp/branch/master/graph/badge.svg?token=XoOTZ0LNPo)](https://codecov.io/gh/libscran/umappp)

## Overview 

**umappp** is a header-only C++ implementation of the Uniform Manifold Approximation and Projection (UMAP) algorithm (McInnes, Healy and Melville, 2018).
UMAP is a non-linear dimensionality reduction technique that is most commonly used for visualization of complex datasets. 
This is achieved by placing each observation on a low-dimensional (usually 2D) embedding in a manner that preserves the neighborhood of each observation from the high-dimensional original space.
The aim is to ensure that the local structure of the data is faithfully recapitulated in lower dimensions 
Further theoretical details can be found in the [original UMAP documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html);
the implementation here is derived from the C++ code in the [**uwot** R package](https://github.com/jlmelville/uwot).

## Quick start

Given a pointer to a column-major input array with `ndim` rows and `nobs` columns, we use `initialize()` to start the UMAP algorithm and `run()` to run it across epochs:

```cpp
#include "umappp/umappp.hpp"

// Assuming `data` contains high-dimensional data in column-major format,
// i.e., each column is a observation and each row is a dimension.
int nrow = 10;
int ncol = 2000;
std::vector<double> data(nrow * ncol);

// Configuring the neighbor search algorithm; here, we'll be using an exact
// search based on VP trees with a Euclidean distance metric.
knncolle::VptreeBuilder<int, double, double> vp_builder(
    std::make_shared<knncolle::EuclideanDistance<double, double> >()
);

// Set number of dimensions in the output embedding.
size_t out_dim = 2;
std::vector<double> embedding(npts * out_dim);

// Initialize the UMAP state:
umappp::Options opt;
auto status = umappp::initialize(
    ndim,
    nobs,
    data.data(),
    vp_builder, 
    out_dim,
    embedding.data(),
    opt
);

// Run UMAP algorithm to completion. This updates the contents
// of the 'embedding' vector supplied to initialize().
status.run();
```

We can modify parameters in the `Options` class that is passed to `initialize()`:

```cpp
opt.num_neighbors = 20;
opt.num_epochs = 200;
opt.min_dist = 0.2;
```

We can also run the algorithm up to the specified number of epochs,
which is occasionally useful for inspecting the intermediate states of the embedding:

```cpp
auto status2 = umappp::initialize(
    ndim,
    nobs,
    data.data(),
    vp_builder,
    out_dim,
    embedding.data(),
    opt
);

for (int iter = 10; iter < 200; iter += 10) {
    status2.run(iter);
    // do something with the current contents of 'embedding',
    // e.g., create an animation over iterations.
}
```

Advanced users can control the neighbor search by either providing the search results directly (as a vector of vectors of index-distance pairs)
or by providing an appropriate [**knncolle**](https://github.com/knncolle/knncolle) subclass to the `initialize()` function:

```cpp
knncolle_annoy::AnnoyBuilder<int, double, double, Annoy::Euclidean> annoy_builder;
auto annoy_idx = annoy_builder.build_unique(knncolle::SimpleMatrix(ndim, nobs, data.data()));
auto status_annoy = umappp::initialize(*annoy_idx, 2, embedding.data(), opt);
status_annoy.run();
```

See the [reference documentation](https://libscran.github.io/umappp) for more details.

## Building projects

### CMake with `FetchContent`

If you're already using CMake, you can add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  umappp 
  GIT_REPOSITORY https://github.com/libscran/umappp
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(umappp)
```

And then:

```cmake
# For executables:
target_link_libraries(myexe libscran::umappp)

# For libaries
target_link_libraries(mylib INTERFACE libscran::umappp)
```

### CMake with `find_package()`

```cmake
find_package(libscran_umappp CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE libscran::umappp)
```

To install the library, use:

```sh
mkdir build && cd build
cmake .. -DUMAPPP_TESTS=OFF
cmake --build . --target install
```

By default, this will use `FetchContent` to fetch all external dependencies.
If you want to install them manually, use `-DUMAPPP_FETCH_EXTERN=OFF`.
See [`extern/CMakeLists.txt`](extern/CMakeLists.txt) to find compatible versions of each dependency.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt), which also need to be made available during compilation.

## References

McInnes L, Healy J, Melville J (2020).
UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
_arXiv_, https://arxiv.org/abs/1802.03426
