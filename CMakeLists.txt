# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(numba_cuda_mlir LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ALLOW_WARNINGS "Don't treat warnings as errors" OFF)
option(
    NUMBA_CUDA_MLIR_PY_GIL_DISABLED
    "Build Python extension modules for a free-threaded CPython runtime"
    OFF
)
if (NOT ALLOW_WARNINGS)
    if(MSVC)
        add_compile_options(/WX)
    else()
        add_compile_options(-Werror)
    endif()
endif()

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_custom_target(devinstall
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE:_cext>"
            "${CMAKE_SOURCE_DIR}/src/numba_cuda_mlir/$<TARGET_FILE_NAME:_cext>")

if (DLPACK_PATH)
    set(dlpack_INCLUDE_DIR "${DLPACK_PATH}/include/dlpack")
    message(STATUS "Using local dlpack, include path: ${dlpack_INCLUDE_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        dlpack
        GIT_REPOSITORY https://github.com/dmlc/dlpack.git
        GIT_TAG v1.1
    )
    FetchContent_Populate(dlpack)
    set(dlpack_INCLUDE_DIR "${dlpack_SOURCE_DIR}/include/dlpack")
    message(STATUS "Fetched dlpack headers from ${dlpack_SOURCE_DIR}")
endif()

add_subdirectory(cext)

option(BUILD_LLVM70 "Build mlir-llvm70 (requires MLIR_DIR)" OFF)
if(BUILD_LLVM70)
  if(NOT DEFINED MLIR_DIR)
    message(FATAL_ERROR "BUILD_LLVM70=ON requires MLIR_DIR to be set")
  endif()
  add_subdirectory(cext/mlir-llvm70)
endif()
