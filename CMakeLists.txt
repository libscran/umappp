cmake_minimum_required(VERSION 3.14...3.25)

project(umappp
    VERSION 2.0.1
    DESCRIPTION "A C++ implementation of the UMAP algorithm"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(umappp INTERFACE)
add_library(libscran::umappp ALIAS umappp)

target_include_directories(umappp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libscran>)
target_compile_features(umappp INTERFACE cxx_std_17)

# Dependencies
option(UMAPPP_FETCH_EXTERN "Automatically fetch umappp's external dependencies." ON)
if(UMAPPP_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(ltla_aarand CONFIG REQUIRED)
    find_package(ltla_irlba CONFIG REQUIRED)
    find_package(ltla_subpar 0.3.1 CONFIG REQUIRED)
    find_package(knncolle_knncolle 2.3.0 CONFIG REQUIRED)
endif()

target_link_libraries(umappp INTERFACE ltla::aarand ltla::irlba ltla::subpar knncolle::knncolle)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(UMAPPP_TESTS "Build umappp's test suite." ON)
else()
    option(UMAPPP_TESTS "Build umappp's test suite." OFF)
endif()
if(UMAPPP_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libscran)

install(TARGETS umappp
    EXPORT umapppTargets)

install(EXPORT umapppTargets
    FILE libscran_umapppTargets.cmake
    NAMESPACE libscran::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_umappp)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_umapppConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_umappp)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_umapppConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libscran_umapppConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_umappp)
