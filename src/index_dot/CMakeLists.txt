cmake_minimum_required(VERSION 3.28)

project(COPY LANGUAGES CXX CUDA)
# add_compile_options(-G)

set(CUTE_INDEX_DOT cute_index_dot)


add_executable(${CUTE_INDEX_DOT} index_dot.cu)

target_include_directories(${CUTE_INDEX_DOT} PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_UTILS_INCLUDE_DIR}
)

set_target_properties(${CUTE_INDEX_DOT} PROPERTIES
    CUDA_ARCHITECTURES 75
)

target_compile_options(${CUTE_INDEX_DOT} PRIVATE
    --expt-relaxed-constexpr
    --extended-lambda
)

find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include CACHE PATH "CUTLASS Header Library")

find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

target_link_libraries(${CUTE_INDEX_DOT} PRIVATE CUDA::cudart)
