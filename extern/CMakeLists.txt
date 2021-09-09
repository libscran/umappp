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

FetchContent_Declare(
  Spectra 
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_TAG v1.0.0
)

FetchContent_MakeAvailable(Spectra)
