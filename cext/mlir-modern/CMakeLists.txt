# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)
project(MLIRModernToNVVM LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(MLIRModernToNVVM SHARED
  ModernBridge.cpp
)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

target_include_directories(MLIRModernToNVVM
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_compile_options(MLIRModernToNVVM PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_libraries(MLIRModernToNVVM
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRBuiltinToLLVMIRTranslation
    MLIRGPUDialect
    MLIRGPUToLLVMIRTranslation
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    LLVMCore
    LLVMBitWriter
    LLVMSupport
)

if(MSVC)
  target_compile_options(MLIRModernToNVVM PRIVATE /wd4244 /wd4267 /wd4706)
endif()

set_target_properties(MLIRModernToNVVM PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(MLIRModernToNVVMSmoke
  ModernBridgeSmoke.cpp
)
target_link_libraries(MLIRModernToNVVMSmoke PRIVATE MLIRModernToNVVM)
set_target_properties(MLIRModernToNVVMSmoke PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

enable_testing()
add_test(NAME MLIRModernToNVVMSmoke COMMAND MLIRModernToNVVMSmoke)

install(TARGETS MLIRModernToNVVM
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
