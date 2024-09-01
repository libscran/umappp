include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

option(CODE_COVERAGE "Enable coverage testing" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

add_executable(
    libtest 
    src/spectral_init.cpp
    src/combine_neighbor_sets.cpp
    src/neighbor_similarities.cpp
    src/optimize_layout.cpp
    src/find_ab.cpp
    src/umappp.cpp
)

macro(decorate_executable name)
    target_link_libraries(
        ${name}
        gtest_main
        umappp
    )

    if(DO_CODE_COVERAGE)
        target_compile_options(${name} PRIVATE -O0 -g --coverage)
        target_link_options(${name} PRIVATE --coverage)
    endif()

    target_compile_options(${name} PRIVATE -Wall -Werror -Wpedantic -Wextra)

    gtest_discover_tests(${name})
endmacro()

decorate_executable(libtest)

# Test the custom parallelization capability.
add_executable(
    cuspartest 
    src/umappp.cpp
    src/spectral_init.cpp
    src/optimize_layout.cpp
)

decorate_executable(cuspartest)
target_compile_definitions(cuspartest PRIVATE TEST_CUSTOM_PARALLEL=1)
