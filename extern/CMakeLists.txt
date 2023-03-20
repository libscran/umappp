include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG 84d48b65d49ce8b844398f11aff3015b86e17197
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/moritz-h/knncolle
  GIT_TAG cmake
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG 0eee555f27aedefad4bd164d7fc93610aed54e13
)

FetchContent_MakeAvailable(irlba)
