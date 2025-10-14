cmake_minimum_required(VERSION 3.28)

project(COPY LANGUAGES CXX CUDA)
# add_compile_options(-G)
add_compile_options(--generate-line-info --keep)
set(CUTE_CCE_FRWD cute_cce_frwd)


add_executable(${CUTE_CCE_FRWD} cce_frwd.cu)

target_include_directories(${CUTE_CCE_FRWD} PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_UTILS_INCLUDE_DIR}
)

set_target_properties(${CUTE_CCE_FRWD} PROPERTIES
    CUDA_ARCHITECTURES 75
)

target_compile_options(${CUTE_CCE_FRWD} PRIVATE
    --expt-relaxed-constexpr
    --extended-lambda
)

find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include CACHE PATH "CUTLASS Header Library")

find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

target_link_libraries(${CUTE_CCE_FRWD} PRIVATE CUDA::cudart)
