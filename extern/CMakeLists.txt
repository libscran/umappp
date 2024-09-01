include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master # ^1.0.0
)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master # ^0.3.1
)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/knncolle/knncolle
  GIT_TAG master # ^2.3.0
)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG master # ^2.0.0
)

FetchContent_MakeAvailable(aarand)
FetchContent_MakeAvailable(subpar)
FetchContent_MakeAvailable(knncolle)
FetchContent_MakeAvailable(irlba)
