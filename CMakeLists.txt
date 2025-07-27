cmake_minimum_required(VERSION 3.20.0)

project(swnumericlib CXX)
add_library(swnumeric INTERFACE)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Set default build type only for main project
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Debug
        CACHE STRING "Choose the type of build." FORCE)
  endif()

  enable_testing()

  # Set default build type to Debug if not specified
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Debug
        CACHE STRING "Choose the type of build." FORCE)
  endif()

  set(CMAKE_CXX_FLAGS_RELEASE
      "-O3 -larmpl -march=native -Wall -Rpass-missed=loop-vectorize")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Applying Debug flags for macOS leaks")
    set(CMAKE_CXX_FLAGS_DEBUG
        "-g -O2 -fno-omit-frame-pointer \
      -Wall -Rpass-missed=loop-vectorize"
        CACHE STRING "" FORCE)
  endif()

  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

  target_include_directories(swnumeric INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(USE_LINALG_BLAS On)

if(USE_LINALG_BLAS)
  message(STATUS "Using BLAS Libs")
  target_compile_definitions(swnumeric INTERFACE USE_LINALG_BLAS=1)

  # Set ARMPL paths (adjust as needed)
  set(ARMPL_ROOT "/opt/arm/armpl_24.10_flang-new_clang_19")
  set(ARMPL_INCLUDE_DIR "${ARMPL_ROOT}/include")
  set(ARMPL_LIB_DIR "${ARMPL_ROOT}/lib")

  # Locate ARMPL libraries
  find_library(
    ARMPL_BLAS_LIB
    NAMES armpl
    PATHS ${ARMPL_LIB_DIR})
  find_library(
    ARMPL_LAPACK_LIB
    NAMES armpl
    PATHS ${ARMPL_LIB_DIR})
  find_library(
    ARMPL_AMATH_LIB
    NAMES amath
    PATHS ${ARMPL_LIB_DIR})

  # Ensure ARMPL is found
  if(ARMPL_BLAS_LIB AND ARMPL_LAPACK_LIB)
    message(STATUS "Found ARMPL BLAS: ${ARMPL_BLAS_LIB}")
    message(STATUS "Found ARMPL LAPACK: ${ARMPL_LAPACK_LIB}")
    message(STATUS "Found ARMPL AMATH: ${ARMPL_AMATH_LIB}")

  else()
    message(FATAL_ERROR "ARMPL libraries not found! Check installation.")
  endif()

  target_link_libraries(
    swnumeric INTERFACE ${ARMPL_BLAS_LIB} ${ARMPL_LAPACK_LIB}
                        ${ARMPL_AMATH_LIB})

  target_include_directories(swnumeric INTERFACE ${ARMPL_INCLUDE_DIR})

else()
  message(STATUS "Not using BLAS Libs")
  target_compile_definitions(swnumeric INTERFACE USE_LINALG_BLAS=0)
endif()

add_subdirectory(Libraries)
add_subdirectory(Applications)
