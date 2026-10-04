cmake_minimum_required(VERSION 3.20.0)
project(mdir VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

# Default arguments for lit.
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# An installation holds the driver and its runtime only: the static libraries
# of the dialects and passes are linked into the driver and are not installed.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)

set(MDIR_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(MDIR_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MDIR_SOURCE_DIR}/include)
include_directories(${MDIR_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(lib/Compiler)
option(MDIR_ENABLE_PYTHON "Build the optional Python compilation interface" OFF)
if(MDIR_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(tools)
