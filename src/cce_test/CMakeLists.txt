cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(COPY LANGUAGES CXX CUDA)

# ---- Executable ----
set(CCE_TEST cce_test)
add_executable(${CCE_TEST} cce_test.cu)

# ---- CUTLASS Setup ----
# These can be set globally, but keeping local for clarity
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS ${CMAKE_SOURCE_DIR}/cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include)

set(CUTLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/include CACHE PATH "CUTLASS include directory")
set(CUTLASS_UTILS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include CACHE PATH "CUTLASS utils include directory")

# ---- Torch Setup ----
# Path to your libtorch folder (should contain 'share/cmake/Torch')
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)

# ---- Include directories ----
target_include_directories(${CCE_TEST} PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_UTILS_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# ---- CUDA and compiler flags ----
set_target_properties(${CCE_TEST} PROPERTIES
    CUDA_ARCHITECTURES 75
)

target_compile_options(${CCE_TEST} PRIVATE
    --expt-relaxed-constexpr
    --extended-lambda
    ${TORCH_CXX_FLAGS}
)

# ---- Linking ----
target_link_libraries(${CCE_TEST}
    PRIVATE
    CUDA::cudart
    ${TORCH_LIBRARIES}
)

# ---- Optional: RPATH for Torch runtime ----
set_property(TARGET ${CCE_TEST} PROPERTY
    BUILD_RPATH "$ORIGIN/../../libtorch/lib"
)
