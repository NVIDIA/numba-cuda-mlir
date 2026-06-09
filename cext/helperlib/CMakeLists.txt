# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_numba_cuda_mlir_python_module(_helperlib
    SOURCES
        _helpermod.cpp
    MSVC_WARNINGS
        /wd4127
)
