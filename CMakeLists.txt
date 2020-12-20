cmake_minimum_required(VERSION 3.10)
project(shesha LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
include_directories(include)
include_directories(glucose-syrup-4.1)

file(GLOB_RECURSE SOURCES "src/*.cu")
# string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75")

add_executable(shesha ${SOURCES})
# Run make libr in glucose-syrup-4.1/simp folder first
target_link_libraries(shesha ${CMAKE_SOURCE_DIR}/glucose-syrup-4.1/simp/lib.a)

install(TARGETS shesha DESTINATION bin)
