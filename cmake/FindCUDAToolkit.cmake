# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(_CUDAToolkit_find root)
    if (EXISTS "${root}/include/cuda.h")
        message(STATUS "Found cuda.h in ${root}/include")
    else()
        message(STATUS "No cuda.h found in ${root}/include")
        return()
    endif()

    set(CUDAToolkit_INCLUDE_DIR "${root}/include" CACHE PATH "" FORCE)
endfunction()

if (NOT CUDAToolkit_INCLUDE_DIRS)
    message(STATUS "Looking for CUDA Toolkit")
    if (DEFINED CUDAToolkit_ROOT)
        _CUDAToolkit_find("${CUDAToolkit_ROOT}")
    endif()
    if (UNIX)
        _CUDAToolkit_find("/usr/local/cuda")
    endif()
    if (MSVC)
        _CUDAToolkit_find("$ENV{CUDA_PATH}")
    endif()
    if (DEFINED ENV{CUDAToolkit_ROOT})
        _CUDAToolkit_find("$ENV{CUDAToolkit_ROOT}")
    endif()
endif()

find_package_handle_standard_args(CUDAToolkit
    REQUIRED_VARS
        CUDAToolkit_INCLUDE_DIR
)

if(CUDAToolkit_FOUND)
    set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIR})
endif()
