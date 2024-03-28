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
    model/model.onnx.c
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

# copy model file to target destination, if not embedding model into header files
add_custom_command(TARGET ONNX_Test POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/model/model.onnx $<TARGET_FILE_DIR:ONNX_Test>)

