include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG 84d48b65d49ce8b844398f11aff3015b86e17197
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG 3ad6b8cdbd281d78c77390d5a6ded4513bdf3860
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG d23a4c12b95563907cf8ca7584b7bb6625ff886b
)

FetchContent_MakeAvailable(irlba)
