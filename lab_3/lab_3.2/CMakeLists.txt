cmake_minimum_required(VERSION 3.27)
project(lab_3_2_)

set(CMAKE_CXX_STANDARD 20)

set(GCC_COVERAGE_COMPILE_FLAGS "-pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

SET(CMAKE_CXX_FLAGS -pthread)

add_executable(lab_3_2_
        main.cpp)
