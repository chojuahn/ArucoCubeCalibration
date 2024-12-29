cmake_minimum_required(VERSION 3.15)

project(ArucoCubeCalibration)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Ensure CMAKE_CONFIGURATION_TYPES is defined
if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES Debug Release)
endif()

# MSVC Multi-threading set
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

# Set the base output directories
include(set_output_directories.cmake)
if(DEFINED CUSTOM_RUNTIME_OUTPUT_DIR)
    set_output_directories(${CUSTOM_RUNTIME_OUTPUT_DIR})
else()
    set_output_directories()
endif()

# Thirdparty
add_subdirectory(thirdparty)

# Main projects
add_subdirectory(aruco_cube_calibration)
