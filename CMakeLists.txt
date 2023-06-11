cmake_minimum_required(VERSION 3.18)
project(NeuralNetwork)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

add_library(lib_womski neural_network.h neural_network.cc labels.h labels.cc)
target_link_libraries(lib_womski Eigen3::Eigen)

add_executable(womski main.cc)
target_link_libraries(womski lib_womski)

file(GLOB TEST_SRCS "*_test.cc")
