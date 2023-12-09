cmake_minimum_required(VERSION 3.0)
project(cl_test)

add_compile_options(-std=c++14)

add_executable(
    gemm_bib_nn 
    src/gemm_bib_nn.cpp
)

add_executable(
    gemm_bib_nn_m4n8 
    src/gemm_bib_nn_m4n8.cpp
)

add_executable(
    gemm_iib_nn 
    src/gemm_iib_nn.cpp
)

add_executable(
    gemm_iii_nn 
    src/gemm_iii_nn.cpp
)

add_executable(
    gemm_iib_tn 
    src/gemm_iib_tn.cpp
)

include_directories(
    ${CMAKE_SOURCE_DIR}/../third_party/OpenCL-Headers-main/
    ${CMAKE_SOURCE_DIR}/../third_party/OpenCL-CLHPP-main/include/
    ${CMAKE_SOURCE_DIR}/../utils/
)

target_link_libraries(
    gemm_bib_nn 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/../third_party/lib64/libOpenCL.so
    log
)

target_link_libraries(
    gemm_bib_nn_m4n8 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/../third_party/lib64/libOpenCL.so
    log
)

target_link_libraries(
    gemm_iib_nn 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/../third_party/lib64/libOpenCL.so
    log
)

target_link_libraries(
    gemm_iii_nn 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/../third_party/lib64/libOpenCL.so
    log
)

target_link_libraries(
    gemm_iib_tn 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/../third_party/lib64/libOpenCL.so
    log
)
