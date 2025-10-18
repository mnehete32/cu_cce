cmake_minimum_required(VERSION 3.28)
project(src VERSION 0.0.1 LANGUAGES CXX CUDA)

add_subdirectory(index_dot)
add_subdirectory(cce_forward)
add_subdirectory(cce_backward)
