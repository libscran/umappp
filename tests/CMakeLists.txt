include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    libtest 
    src/spectral_init.cpp
    src/combine_neighbor_sets.cpp
    src/neighbor_similarities.cpp
    src/optimize_layout.cpp
    src/find_ab.cpp
    src/Umap.cpp
)

target_link_libraries(
    libtest
    gtest_main
    umappp
)

# Test the custom parallelization capability.
add_executable(
    cuspartest 
    src/Umap.cpp
    src/spectral_init.cpp
    src/optimize_layout.cpp
)

target_link_libraries(
    cuspartest
    gtest_main
    umappp
)

target_compile_definitions(cuspartest PRIVATE TEST_CUSTOM_PARALLEL=1)

set(CODE_COVERAGE "Enable coverage testing" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

# Avoid putting too much stress on CI systems with limited CPUs.
option(TEST_NUM_THREADS "Set the number of threads for testing" 0) 
if(TEST_NUM_THREADS GREATER 0)
    target_compile_definitions(libtest PRIVATE TEST_NUM_THREADS=${TEST_NUM_THREADS})
    target_compile_definitions(cuspartest PRIVATE TEST_NUM_THREADS=${TEST_NUM_THREADS})
endif()

include(GoogleTest)
gtest_discover_tests(libtest)
gtest_discover_tests(cuspartest)
