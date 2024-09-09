cmake_minimum_required(VERSION 3.10)

# Set the project name and specify the programming language (C++)
project(robust_bound CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find all .cpp files in the current directory
file(GLOB SOURCES "*.cpp")

# Include the current directory for header files
include_directories(${CMAKE_SOURCE_DIR})

# Add an executable using all .cpp files
add_executable(robust_bound ${SOURCES})

