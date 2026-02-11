cmake_minimum_required(VERSION 3.20)

project(SWNUMERIC LANGUAGES CXX)

# Specify the C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(swnumeric_lib STATIC dummy.cpp)
target_include_directories(swnumeric_lib PUBLIC ${CMAKE_SOURCE_DIR})

find_package(MKL REQUIRED)
target_link_libraries(swnumeric_lib PUBLIC MKL::MKL)


add_subdirectory(library)
add_subdirectory(routines)

