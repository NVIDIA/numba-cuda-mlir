# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(_cext MODULE
    check.cpp
    cuda_loader.cpp
    cuda_helper.cpp
    llvm_downgrade.cpp
    module.cpp
    py.cpp
    kernel.cpp
    )

target_compile_options(_cext PUBLIC -fno-exceptions -fno-rtti)

target_include_directories(_cext PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/dlpack"
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    )

set_target_properties(_cext PROPERTIES SKIP_BUILD_RPATH TRUE)

if (APPLE)
    target_link_options(_cext PRIVATE -undefined dynamic_lookup)
endif()

