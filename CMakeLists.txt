cmake_minimum_required(VERSION 3.16)
project(DeACpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

find_path(OPENCV_INCLUDE_DIR opencv2/core.hpp
    PATHS
        /usr/include/opencv4
        /usr/local/include/opencv4
        /home/gd60v1/.local/include/opencv4
)

if(NOT OPENCV_INCLUDE_DIR)
    message(FATAL_ERROR "OpenCV headers not found (opencv2/core.hpp).")
endif()

set(OPENCV_REQUIRED_LIB_NAMES
    opencv_gapi
    opencv_highgui
    opencv_ml
    opencv_objdetect
    opencv_photo
    opencv_stitching
    opencv_video
    opencv_calib3d
    opencv_features2d
    opencv_dnn
    opencv_flann
    opencv_videoio
    opencv_imgcodecs
    opencv_imgproc
    opencv_core
)

set(OPENCV_LIBRARIES "")
foreach(OPENCV_LIB_NAME IN LISTS OPENCV_REQUIRED_LIB_NAMES)
    unset(OPENCV_LIB_PATH CACHE)
    find_library(OPENCV_LIB_PATH NAMES ${OPENCV_LIB_NAME})
    if(NOT OPENCV_LIB_PATH)
        message(FATAL_ERROR "Required OpenCV library not found: ${OPENCV_LIB_NAME}")
    endif()
    list(APPEND OPENCV_LIBRARIES ${OPENCV_LIB_PATH})
endforeach()

add_executable(deacpp
    src/main.cpp
)

target_include_directories(deacpp PRIVATE
    ${OPENCV_INCLUDE_DIR}
)

target_compile_definitions(deacpp PRIVATE
    DEA_CPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(deacpp PRIVATE
    ${OPENCV_LIBRARIES}
    Threads::Threads
)

# TensorRT is optional. If found, .engine loading is enabled.
find_path(NVINFER_INCLUDE_DIR NvInfer.h
    PATHS
        /usr/include/aarch64-linux-gnu
        /usr/include/x86_64-linux-gnu
        /usr/include
)
find_library(NVINFER_LIB nvinfer)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin)
find_library(NVONNXPARSER_LIB nvonnxparser)
find_library(CUDART_LIB cudart
    PATHS
        /usr/local/cuda/lib64
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
)
find_path(CUDA_INCLUDE_DIR cuda_fp16.h
    PATHS
        /usr/local/cuda/include
        /usr/include
)

if(NVINFER_INCLUDE_DIR AND NVINFER_LIB AND NVINFER_PLUGIN_LIB)
    message(STATUS "TensorRT found: enabling .engine support")
    target_compile_definitions(deacpp PRIVATE DEA_ENABLE_TENSORRT=1)
    target_include_directories(deacpp PRIVATE ${NVINFER_INCLUDE_DIR})
    if(CUDA_INCLUDE_DIR)
        target_include_directories(deacpp PRIVATE ${CUDA_INCLUDE_DIR})
    endif()
    target_link_libraries(deacpp PRIVATE ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB})
    if(NVONNXPARSER_LIB)
        target_link_libraries(deacpp PRIVATE ${NVONNXPARSER_LIB})
        target_compile_definitions(deacpp PRIVATE DEA_ENABLE_TRT_ONNX_PARSER=1)
    endif()
    if(CUDART_LIB)
        target_link_libraries(deacpp PRIVATE ${CUDART_LIB})
    endif()
else()
    message(WARNING "TensorRT headers/libs not found. .engine loading disabled.")
endif()

# VPI is optional. If found, --flow-gpu uses VPI OpticalFlowPyrLK on CUDA backend.
find_path(VPI_INCLUDE_DIR vpi/Status.h
    PATHS
        /opt/nvidia/vpi3/include
        /usr/include
)
find_library(VPI_LIB nvvpi
    PATHS
        /opt/nvidia/vpi3/lib/aarch64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
)

if(VPI_INCLUDE_DIR AND VPI_LIB)
    message(STATUS "VPI found: enabling GPU optical flow via VPI")
    target_compile_definitions(deacpp PRIVATE DEA_ENABLE_VPI=1)
    target_include_directories(deacpp PRIVATE ${VPI_INCLUDE_DIR})
    target_link_libraries(deacpp PRIVATE ${VPI_LIB})
else()
    message(STATUS "VPI not found: --flow-gpu will fallback to CPU optical flow")
endif()
