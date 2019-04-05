cmake_minimum_required(VERSION 3.10)
project(sat_maxsat_gpu LANGUAGES CXX CUDA)

include_directories(include)

cuda_add_executable(sat_maxsat_gpu src/main.cu)
