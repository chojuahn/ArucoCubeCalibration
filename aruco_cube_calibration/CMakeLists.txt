
project(calibrate_aruco_cube_camera)

file(GLOB_RECURSE _utils LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/utils/*")
file(GLOB_RECURSE _main LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/*")

# Add executable
add_executable(${PROJECT_NAME}
    ${_utils}
    ${_main}
)

# Link libraries
include_directories(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${PROJECT_NAME}
    cxxopts
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    dlib
    opencv_world
)
if(APPLE)
find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
endif()

# Set properties
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/" FILES ${_utils} ${_main})
set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "AucoCube"
    OUTPUT_NAME ${PROJECT_NAME}
)
