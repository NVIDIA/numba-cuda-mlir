# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_numba_cuda_mlir_python_module(_cext
    SOURCES
        check.cpp
        cuda_loader.cpp
        cuda_helper.cpp
        llvm_downgrade.cpp
        module.cpp
        py.cpp
        kernel.cpp
    INCLUDE_DIRS
        ${dlpack_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)
