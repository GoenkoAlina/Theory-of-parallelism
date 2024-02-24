cmake_minimum_required(VERSION 3.16.3)
add_executable(lab_2.3 lab_2.3.cpp)
target_link_options(lab_2.3 PUBLIC "-fopenmp")