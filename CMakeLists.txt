cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# add_compile_options(-G)
# add_compile_options(--generate-line-info --keep)
find_package(CUDAToolkit REQUIRED)

project(CUTLASS-Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit


set(CUTLASS_NVCC_ARCHS 75 CACHE STRING "CUDA SM architectures for CUTLASS")
set(CUTLASS_ENABLE_TESTS OFF CACHE STRING "CUDA TURN OF TEST CUTLASS")


# Set CUTLASS include directories
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include CACHE PATH "CUTLASS Header Library")

find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

add_subdirectory(src)