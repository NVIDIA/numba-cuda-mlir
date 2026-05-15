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

if(MSVC)
    target_compile_options(_cext PUBLIC /GR- /wd4530)
else()
    target_compile_options(_cext PUBLIC -fno-exceptions -fno-rtti)
endif()

target_include_directories(_cext PRIVATE
    ${dlpack_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    )
if(WIN32)
    target_link_libraries(_cext PRIVATE Python::Module)
endif()

set_target_properties(_cext PROPERTIES SKIP_BUILD_RPATH TRUE)

if (APPLE)
    target_link_options(_cext PRIVATE -undefined dynamic_lookup)
endif()
