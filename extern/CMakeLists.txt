include(FetchContent)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG bd06e0d52e59e35f0456a870746a959da4bc7cb3
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG 560436d142d08eb505fe63fb7eac12ed46fda345
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG 64d7da3e7050ce228b10c8f3057d6d7bcb740593
)

FetchContent_MakeAvailable(irlba)
