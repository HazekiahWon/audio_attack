cmake_minimum_required(VERSION 3.10)
project(myctc)

set(CMAKE_CXX_STANDARD 11)

add_executable(myctc main.cpp)
include_directories(/home/ww/anaconda3/envs/whz_tfgpu/lib/python3.5/site-packages/tensorflow/include/)