# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
cmake_minimum_required(VERSION 3.20)
#===----------------------------------------------------------------------===//
# mlir-llvm70 — MLIR to LLVM70 (LLVM 7 dialect) translator
#
# Required: MLIR_DIR (must match the LLVM used by mlir-python-bindings)
#===----------------------------------------------------------------------===//

set(LLVM70_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LLVM70_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

find_package(MLIR REQUIRED CONFIG)
message(STATUS "mlir-llvm70: MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(LLVM70_COMPILE_OPTIONS)
set(LLVM70_COMPILE_DEFINITIONS)

if(NOT LLVM_ENABLE_RTTI)
  if(MSVC)
    list(APPEND LLVM70_COMPILE_OPTIONS /GR-)
  else()
    list(APPEND LLVM70_COMPILE_OPTIONS -fno-rtti)
  endif()
endif()

if(MSVC)
  list(APPEND LLVM70_COMPILE_OPTIONS /EHsc)
  list(APPEND LLVM70_COMPILE_DEFINITIONS _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING)
  # Windows wheels load MLIR Python and MLIRToLLVM70 as separate DLLs.  MLIR's
  # fallback TypeIDs keep registered op/type/interface IDs consistent across
  # those DSOs when both are compiled with this define.
  list(APPEND LLVM70_COMPILE_DEFINITIONS MLIR_USE_FALLBACK_TYPE_IDS=1)
endif()

if(WIN32)
  list(APPEND LLVM70_COMPILE_DEFINITIONS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

function(configure_llvm70_target target)
  target_compile_options(${target} PRIVATE ${LLVM70_COMPILE_OPTIONS})
  target_compile_definitions(${target} PRIVATE ${LLVM70_COMPILE_DEFINITIONS})
endfunction()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM70_SOURCE_DIR}/include)
include_directories(${LLVM70_BINARY_DIR}/include)

add_subdirectory(include/llvm70/Dialect)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
