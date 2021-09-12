include(FetchContent)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG master
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master
)

FetchContent_MakeAvailable(aarand)

SET(USE_SYSTEM_EIGEN OFF CACHE BOOL "Disable Spectra's use of system Eigen")

FetchContent_Declare(
  Spectra 
  GIT_REPOSITORY https://github.com/LTLA/spectra
  GIT_TAG master
)

FetchContent_MakeAvailable(Spectra)
