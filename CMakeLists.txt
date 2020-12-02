cmake_minimum_required (VERSION 2.8.11)  #cmake版本要求
project (GAMES101)  #命名这个project
include_directories(${PROJECT_SOURCE_DIR})  #引用Eigen, ${PROJECT_SOURCE_DIR}即HF/的绝对路径，代码框架里引用Eigen的方式是<eigen3/Eigen/Eigen>

add_subdirectory(test)
add_subdirectory(Assignment0)
add_subdirectory(Assignment1)
add_subdirectory(Assignment2)
add_subdirectory(Assignment3)