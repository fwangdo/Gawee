# CMakeLists.txt for Gawee Runtime
#
# Build instructions:
#   mkdir build && cd build
#   cmake ..
#   make
#
# Run:
#   ./gawee_parser ../jsondata/graph.json

cmake_minimum_required(VERSION 3.16)
project(GaweeRuntime VERSION 0.1.0 LANGUAGES CXX)

# Require C++17 for std::filesystem and other modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Source files
set(SOURCES
    src/Graph.cpp
    src/Parser.cpp
    src/main.cpp
)

# Create executable
add_executable(gawee_parser ${SOURCES})

# Link filesystem library (needed on some systems)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(gawee_parser stdc++fs)
endif()

# Print build info
message(STATUS "Gawee Runtime")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
