cmake_minimum_required(VERSION 3.10)
project(sat_maxsat_gpu LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cu")

add_executable(sat_maxsat_gpu ${SOURCES})

install(TARGETS sat_maxsat_gpu DESTINATION bin)
