# Enable FetchContent module
include(FetchContent)

# ===============================
# Fetch Json
# ===============================
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(json)

# ===============================
# Fetch Cxxopts
# ===============================
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.0
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(cxxopts)

# ===============================
# Fetch Eigen
# ===============================
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    EXCLUDE_FROM_ALL
    SOURCE_SUBDIR cmake             # Hack: causing not to generate eigen, supplying cmake as dummy path
)
FetchContent_MakeAvailable(eigen)   # Won't generate eigne
add_library(eigen INTERFACE)        # Instead, make an interface to eigen
target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})

# ===============================
# Fetch dlib
# ===============================
FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_SHALLOW TRUE
    GIT_TAG v19.24
)
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
set(DLIB_JPEG_SUPPORT OFF CACHE INTERNAL "")
set(DLIB_LINK_WITH_SQLITE3 OFF CACHE INTERNAL "")
set(DLIB_USE_CUDA OFF CACHE INTERNAL "")
set(DLIB_PNG_SUPPORT OFF CACHE INTERNAL "")
set(DLIB_NO_GUI_SUPPORT ON CACHE INTERNAL "")
FetchContent_MakeAvailable(dlib)
set_target_properties(dlib PROPERTIES FOLDER "Thirdparty")

# ===============================
# Fetch OpenCV
# ===============================
FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.10.0
    GIT_SHALLOW 1
)
# Core modules
set(BUILD_opencv_calib3d             ON  CACHE INTERNAL "")
set(BUILD_opencv_core                ON  CACHE INTERNAL "")
set(BUILD_opencv_features2d          ON  CACHE INTERNAL "")
set(BUILD_opencv_flann               ON  CACHE INTERNAL "")
set(BUILD_opencv_highgui             ON  CACHE INTERNAL "")
set(BUILD_opencv_imgcodecs           ON  CACHE INTERNAL "")
set(BUILD_opencv_imgproc             ON  CACHE INTERNAL "")
set(BUILD_opencv_objdetect           ON  CACHE INTERNAL "")
set(BUILD_opencv_video               ON  CACHE INTERNAL "")
set(BUILD_opencv_videoio             ON  CACHE INTERNAL "")
set(BUILD_opencv_world               ON  CACHE INTERNAL "")

# Newer (OpenCV 4.x) modules
# If you do not need G-API, you can set it OFF
set(BUILD_opencv_gapi                OFF CACHE INTERNAL "")

# Disabled OpenCV modules
set(BUILD_opencv_java_bindings_generator OFF CACHE INTERNAL "")
set(BUILD_opencv_js                       OFF CACHE INTERNAL "")
set(BUILD_opencv_python_bindings_generator OFF CACHE INTERNAL "")
set(BUILD_opencv_cudaarithm               OFF CACHE INTERNAL "")
set(BUILD_opencv_cudabgsegm               OFF CACHE INTERNAL "")
set(BUILD_opencv_cudacodec                OFF CACHE INTERNAL "")
set(BUILD_opencv_cudafeatures2d           OFF CACHE INTERNAL "")
set(BUILD_opencv_cudafilters              OFF CACHE INTERNAL "")
set(BUILD_opencv_cudaimgproc              OFF CACHE INTERNAL "")
set(BUILD_opencv_cudalegacy               OFF CACHE INTERNAL "")
set(BUILD_opencv_cudaobjdetect            OFF CACHE INTERNAL "")
set(BUILD_opencv_cudaoptflow              OFF CACHE INTERNAL "")
set(BUILD_opencv_cudastereo               OFF CACHE INTERNAL "")
set(BUILD_opencv_cudawarping              OFF CACHE INTERNAL "")
set(BUILD_opencv_cudev                    OFF CACHE INTERNAL "")
set(BUILD_opencv_dnn                      OFF CACHE INTERNAL "")
set(BUILD_opencv_java                     OFF CACHE INTERNAL "")
set(BUILD_opencv_ml                       OFF CACHE INTERNAL "")
set(BUILD_opencv_photo                    OFF CACHE INTERNAL "")
set(BUILD_opencv_python2                  OFF CACHE INTERNAL "")
set(BUILD_opencv_python3                  OFF CACHE INTERNAL "")
set(BUILD_opencv_shape                    OFF CACHE INTERNAL "")
set(BUILD_opencv_stitching                OFF CACHE INTERNAL "")
set(BUILD_opencv_superres                 OFF CACHE INTERNAL "")
set(BUILD_opencv_ts                       OFF CACHE INTERNAL "")
set(BUILD_opencv_videostab                OFF CACHE INTERNAL "")
set(BUILD_opencv_viz                      OFF CACHE INTERNAL "")

# --------------------------------------------------------------
# OpenCV build options
# --------------------------------------------------------------
set(BUILD_SHARED_LIBS           OFF CACHE INTERNAL "")  # Statically link
set(BUILD_opencv_apps           OFF CACHE INTERNAL "")
set(BUILD_PROTOBUF              OFF CACHE INTERNAL "")
set(BUILD_ANDROID_EXAMPLES      OFF CACHE INTERNAL "")
set(BUILD_DOCS                  OFF CACHE INTERNAL "")
set(BUILD_EXAMPLES              OFF CACHE INTERNAL "")
set(BUILD_PACKAGE               OFF CACHE INTERNAL "")
set(BUILD_PERF_TESTS            OFF CACHE INTERNAL "")
set(BUILD_TESTS                 OFF CACHE INTERNAL "")
set(BUILD_WITH_DYNAMIC_IPP      OFF CACHE INTERNAL "")
set(BUILD_FAT_JAVA_LIB          OFF CACHE INTERNAL "")
set(BUILD_ANDROID_SERVICE       OFF CACHE INTERNAL "")
set(BUILD_WITH_STATIC_CRT       OFF CACHE INTERNAL "")

# --------------------------------------------------------------
# 3rdparty dependencies / additional features
# --------------------------------------------------------------
set(BUILD_ZLIB          ON  CACHE INTERNAL "")    # Zlib is widely used
set(WITH_CUDA           OFF CACHE INTERNAL "")
set(WITH_VFW            ON  CACHE INTERNAL "")    # Video for Windows

# Windows-specific camera/video options
if (WIN32)
set(WITH_DSHOW     ON  CACHE INTERNAL "")     # DirectShow for UVC cameras
set(WITH_WIN32UI   ON  CACHE INTERNAL "")
else()
set(WITH_DSHOW     OFF CACHE INTERNAL "")
set(WITH_WIN32UI   OFF CACHE INTERNAL "")
endif()

# macOS-specific toggles
if (APPLE)
# Often we rely on macOS native image codecs
set(WITH_JPEG     OFF CACHE INTERNAL "")
else()
set(WITH_JPEG     ON  CACHE INTERNAL "")
endif()

# Other optional 3rdparty libs
set(WITH_OBSENSOR      OFF CACHE INTERNAL "")
set(WITH_GSTREAMER     OFF CACHE INTERNAL "")
set(WITH_WEBP          OFF CACHE INTERNAL "")
set(WITH_VTK           OFF CACHE INTERNAL "")
set(WITH_TIFF          OFF CACHE INTERNAL "")
set(WITH_PNG           OFF CACHE INTERNAL "")
set(WITH_OPENEXR       OFF CACHE INTERNAL "")
set(WITH_OPENCLAMDBLAS OFF CACHE INTERNAL "")
set(WITH_OPENCLAMDFFT  OFF CACHE INTERNAL "")
set(WITH_MATLAB        OFF CACHE INTERNAL "")
set(WITH_LAPACK        OFF CACHE INTERNAL "")
set(WITH_JASPER        OFF CACHE INTERNAL "")
set(WITH_IPP           OFF CACHE INTERNAL "")
set(WITH_FFMPEG        ON  CACHE INTERNAL "")
IF (WIN32)
set(WITH_MSMF          ON  CACHE INTERNAL "")
endif()
set(WITH_DIRECTX       OFF CACHE INTERNAL "")
set(WITH_CUFFT         OFF CACHE INTERNAL "")
set(WITH_1394          OFF CACHE INTERNAL "")
set(WITH_OPENCL        OFF CACHE INTERNAL "")
set(WITH_ITT           OFF CACHE INTERNAL "")
set(WITH_EIGEN         OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(opencv)

target_include_directories(opencv_world INTERFACE
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/core/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/flann/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/imgproc/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/features2d/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/imgcodecs/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/videoio/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/calib3d/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/highgui/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/objdetect/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/video/include>
    $<BUILD_INTERFACE:${OpenCV_SOURCE_DIR}/modules/world/include>
    $<BUILD_INTERFACE:${OpenCV_BINARY_DIR}/modules/world>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}> # OpenCV generates a opencv_modules.hpp file in the output dir
)
set_target_properties(ade PROPERTIES FOLDER "Thirdparty")
set_target_properties(libopenjp2 PROPERTIES FOLDER "Thirdparty")
set_target_properties(zlib PROPERTIES FOLDER "Thirdparty")
IF (WIN32)
set_target_properties(jpeg12-static PROPERTIES FOLDER "Thirdparty")
set_target_properties(jpeg16-static PROPERTIES FOLDER "Thirdparty")
set_target_properties(libjpeg-turbo PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_world_AVX PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_world_AVX2 PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_world_AVX512_SKX PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_world_SSE4_1 PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_world_SSE4_2 PROPERTIES FOLDER "Thirdparty")
elseif (APPLE)
set_target_properties(carotene PROPERTIES FOLDER "Thirdparty")
set_target_properties(carotene_objs PROPERTIES FOLDER "Thirdparty")
set_target_properties(tegra_hal PROPERTIES FOLDER "Thirdparty")
endif()
set_target_properties(opencv_world PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_modules PROPERTIES FOLDER "Thirdparty")
set_target_properties(gen_opencv_js_source PROPERTIES FOLDER "Thirdparty")
set_target_properties(gen_opencv_objc_source PROPERTIES FOLDER "Thirdparty")
set_target_properties(gen_opencv_objc_source_ios PROPERTIES FOLDER "Thirdparty")
set_target_properties(gen_opencv_objc_source_osx PROPERTIES FOLDER "Thirdparty")
set_target_properties(gen_opencv_objc_source_visionos PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_dnn_plugins PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_highgui_plugins PROPERTIES FOLDER "Thirdparty")
set_target_properties(opencv_videoio_plugins PROPERTIES FOLDER "Thirdparty")
