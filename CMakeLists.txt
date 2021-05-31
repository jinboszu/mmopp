cmake_minimum_required(VERSION 3.16)
project(mmopp CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic")

include_directories(.)

add_executable(solve solve.cpp)
