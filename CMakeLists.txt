cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(ONNX_Test)

# Requirements for MacOS
find_library(FOUNDATION Foundation)
find_library(COREFOUNDATION CoreFoundation)


add_executable(
    ONNX_Test 
    onnx_test.cpp 
    model/model.ort.c
)

target_link_libraries(
    ONNX_Test 
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/lib/libonnxruntime.a
    ${CoreFoundation}
    ${Foundation}
)

target_include_directories(
    ONNX_Test 
    PUBLIC
    ./include
)

