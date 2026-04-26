cmake_minimum_required(VERSION 3.20)
project(gawee_back LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gawee-aot gawee_aot.cpp)
target_include_directories(gawee-aot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gawee-eval gawee_eval.cpp)
target_include_directories(gawee-eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
