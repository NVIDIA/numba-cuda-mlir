# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(add_numba_cuda_mlir_python_module target)
    cmake_parse_arguments(
        NC_EXT
        ""
        ""
        "SOURCES;INCLUDE_DIRS;MSVC_WARNINGS"
        ${ARGN}
    )

    add_library(${target} MODULE ${NC_EXT_SOURCES})

    if(MSVC)
        target_compile_options(${target} PUBLIC /GR- /wd4530 ${NC_EXT_MSVC_WARNINGS})
    else()
        target_compile_options(${target} PUBLIC -fno-exceptions -fno-rtti)
    endif()

    target_include_directories(${target} PRIVATE
        ${Python_INCLUDE_DIRS}
        ${NC_EXT_INCLUDE_DIRS}
    )

    if(WIN32)
        target_link_libraries(${target} PRIVATE Python::Module)
    endif()

    set_target_properties(${target} PROPERTIES SKIP_BUILD_RPATH TRUE)

    if(APPLE)
        target_link_options(${target} PRIVATE -undefined dynamic_lookup)
    endif()
endfunction()

add_subdirectory(launcher)
add_subdirectory(typeconv)
add_subdirectory(mviewbuf)
add_subdirectory(helperlib)
