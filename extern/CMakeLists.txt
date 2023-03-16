include(FetchContent)

FetchContent_Declare(
  knncolle 
  GIT_REPOSITORY https://github.com/LTLA/knncolle
  GIT_TAG c5a1776ebf10641d9bf8715e89cb2d965b06e899
)

FetchContent_MakeAvailable(knncolle)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG e8397e9e648379f6086b20a83910c39ed8e4dd45
)

FetchContent_MakeAvailable(aarand)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG f869614019048f84008c18df364b41ebbf2303dd
)

FetchContent_MakeAvailable(irlba)
