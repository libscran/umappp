cmake_minimum_required(VERSION 3.14)

project(umappp
    VERSION 1.0.0
    DESCRIPTION "A C++ implementation of the UMAP algorithm"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(umappp INTERFACE)

target_include_directories(umappp INTERFACE include/)

add_subdirectory(extern)

target_link_libraries(umappp INTERFACE knncolle aarand irlba)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
