cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(CUTLASS_Examples VERSION 0.0.1 LANGUAGES CXX CUDA)

# ---- Global compiler standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- CUDA Setup ----
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.6)

find_package(CUDAToolkit REQUIRED)

# ---- LibTorch Setup ----
# Make sure this path contains 'share/cmake/Torch'
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---- CUTLASS Setup ----
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include CACHE PATH "CUTLASS include directory")
set(CUTLASS_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include CACHE PATH "CUTLASS utils include directory")

include_directories(
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_UTILS_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ---- CUDA Compile Options ----
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# ---- Global Link Libraries ----
link_libraries(${TORCH_LIBRARIES})

# ---- CUTLASS Options ----
set(CUTLASS_NVCC_ARCHS 75 CACHE STRING "CUDA SM architectures for CUTLASS")
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests")

# ---- Add all source subdirectories ----
add_subdirectory(src)

# ---- Optional: Runtime library path for Torch ----
set(CMAKE_INSTALL_RPATH "$ORIGIN/libtorch/lib")
set(CMAKE_BUILD_RPATH "$ORIGIN/libtorch/lib")
