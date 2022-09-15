cmake_minimum_required(VERSION 3.15)
project(tvm-mlir LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options(/wd4005 /wd4251 /wd4530)
endif(MSVC)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

find_package(MLIR REQUIRED CONFIG)
find_package(fmt REQUIRED)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${PROJECT_BINARY_DIR}/lib)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
