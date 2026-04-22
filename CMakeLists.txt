# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(cusimt LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ALLOW_WARNINGS "Don't treat warnings as errors" OFF)
if (NOT ALLOW_WARNINGS)
    add_compile_options(-Werror)
endif()

add_compile_options(-Wall)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

add_custom_target(devinstall
    COMMAND ln -fs "${CMAKE_BINARY_DIR}/cext/launcher/lib_cext.so"
            ${CMAKE_SOURCE_DIR}/src/cusimt/_cext.so)

add_subdirectory(cext)

option(BUILD_NVVM70 "Build mlir-nvvm70 (requires MLIR_DIR)" OFF)
if(BUILD_NVVM70)
  if(NOT DEFINED MLIR_DIR)
    message(FATAL_ERROR "BUILD_NVVM70=ON requires MLIR_DIR to be set")
  endif()
  add_subdirectory(cext/mlir-nvvm70)
endif()
