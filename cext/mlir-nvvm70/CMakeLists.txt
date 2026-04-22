# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
cmake_minimum_required(VERSION 3.20)
#===----------------------------------------------------------------------===//
# mlir-nvvm70 — MLIR to NVVM70 (LLVM 7 dialect) translator
#
# Required: MLIR_DIR (must match the LLVM used by mlir-python-bindings)
#===----------------------------------------------------------------------===//

set(NVVM70_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(NVVM70_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

find_package(MLIR REQUIRED CONFIG)
message(STATUS "mlir-nvvm70: MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${NVVM70_SOURCE_DIR}/include)
include_directories(${NVVM70_BINARY_DIR}/include)

add_subdirectory(include/nvvm70/Dialect)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
