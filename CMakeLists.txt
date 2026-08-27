cmake_minimum_required(VERSION 3.16)
project(rv_sparse C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

# The benchmark builds one tree per (march, cflags) variant and passes the ISA
# string in here, so it is a cache variable rather than baked into the project.
set(RVSP_ARCH_FLAGS "-march=native" CACHE STRING
    "Target architecture flags, e.g. -march=rv64gcv -mabi=lp64d")

option(RVSP_OPENMP "Build the OpenMP kernel" OFF)
option(RVSP_LTO "Link-time optimisation in release builds" ON)
option(RVSP_BUILD_TESTS "Build the test programs" ON)
option(RVSP_BUILD_EXAMPLES "Build the example programs" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

separate_arguments(RVSP_ARCH_FLAG_LIST NATIVE_COMMAND "${RVSP_ARCH_FLAGS}")

add_compile_options(-Wall -Wextra -Wpedantic ${RVSP_ARCH_FLAG_LIST})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)

    if(RVSP_LTO)
        add_compile_options(-flto)
        add_link_options(-flto)
    else()
        message(STATUS "LTO disabled")
    endif()
else()
    add_compile_options(-g -O0)
    add_compile_definitions(DEBUG)
endif()

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

file(GLOB_RECURSE RVSP_SRCS CONFIGURE_DEPENDS src/*.c)

# Legacy standalone prototype; it may carry its own main().
list(FILTER RVSP_SRCS EXCLUDE REGEX "src/matmul/AxBRowIP\\.c$")

# Vector kernels #error without the V extension rather than silently falling
# back to scalar, so they are excluded by path when -march does not enable V.
# Match only the RISC-V ISA string so an unrelated flag cannot match "v".
if(RVSP_ARCH_FLAGS MATCHES "(^| )-march=rv[0-9]+[a-z0-9_]*v")
    message(STATUS "vector kernels included")
else()
    message(STATUS "vector kernels excluded: V extension not enabled")
    list(FILTER RVSP_SRCS EXCLUDE REGEX
         "accum_rvv_f32_m[124]\\.c$")
endif()

# The OpenMP kernel includes <omp.h>, which only exists when the toolchain
# ships OpenMP.
if(NOT RVSP_OPENMP)
    message(STATUS "OpenMP kernel excluded: configure with -DRVSP_OPENMP=ON")
    list(FILTER RVSP_SRCS EXCLUDE REGEX "accum_scalar_omp_f32\\.c$")
endif()

add_library(rvsparse STATIC ${RVSP_SRCS})

target_include_directories(rvsparse PUBLIC include)

if(RVSP_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS C)
    target_link_libraries(rvsparse PUBLIC OpenMP::OpenMP_C)
endif()

# ------------------------------------------------------------------------------
# Tools
#
# Matrix generation and Matrix Market loading. The benchmark and the tests link
# these directly rather than through the library.
# ------------------------------------------------------------------------------

file(GLOB RVSP_TOOLS_SRCS CONFIGURE_DEPENDS tools/src/*.c)

add_library(rvsparse_tools STATIC ${RVSP_TOOLS_SRCS})
target_include_directories(rvsparse_tools PUBLIC tools/include)
target_link_libraries(rvsparse_tools PUBLIC rvsparse)

# ------------------------------------------------------------------------------
# Tests and examples
#
# Discovered by glob so a new file needs no build change.
# ------------------------------------------------------------------------------

if(RVSP_BUILD_TESTS)
    enable_testing()

    file(GLOB RVSP_TEST_SRCS CONFIGURE_DEPENDS tests/*.c)

    foreach(src ${RVSP_TEST_SRCS})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE rvsparse_tools m)
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test")
        add_test(NAME ${name} COMMAND ${name})
    endforeach()
endif()

if(RVSP_BUILD_EXAMPLES)
    file(GLOB RVSP_EXAMPLE_SRCS CONFIGURE_DEPENDS examples/*.c)

    foreach(src ${RVSP_EXAMPLE_SRCS})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE rvsparse_tools m)
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")
    endforeach()
endif()
