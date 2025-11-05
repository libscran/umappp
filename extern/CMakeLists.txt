include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master # ^1.0.0
)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master # ^0.4.0
)

FetchContent_Declare(
  eigen 
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG 5.0.0
)

FetchContent_Declare(
  sanisizer
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master # ^0.1.3
)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/knncolle/knncolle
  GIT_TAG master # ^3.0.0
)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG master # ^2.0.0
)

FetchContent_MakeAvailable(aarand)
FetchContent_MakeAvailable(subpar)
FetchContent_MakeAvailable(sanisizer)
FetchContent_MakeAvailable(knncolle)

# Forcing the Eigen build to create the installation files,
# otherwise scran_pca's own installation gets confused.
option(EIGEN_BUILD_CMAKE_PACKAGE "Build the Eigen Cmake package" ON)
FetchContent_MakeAvailable(eigen)

FetchContent_MakeAvailable(irlba)
