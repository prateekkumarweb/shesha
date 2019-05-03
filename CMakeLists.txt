cmake_minimum_required(VERSION 3.10)
project(sat_maxsat_gpu LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
include_directories(include)
include_directories(glucose-syrup-4.1)

file(GLOB_RECURSE SOURCES "src/*.cu")

add_executable(sat_maxsat_gpu ${SOURCES})
# Run make libr in glucose-syrup-4.1/simp folder first
target_link_libraries(sat_maxsat_gpu ${CMAKE_SOURCE_DIR}/glucose-syrup-4.1/simp/lib.a)

install(TARGETS sat_maxsat_gpu DESTINATION bin)
