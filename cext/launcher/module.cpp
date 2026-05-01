/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "py.h"

#include "cuda_loader.h"
#include "kernel.h"
#include "cuda_helper.h"
#include "llvm_downgrade.h"


static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "numba_cuda_mlir._cext",
    .m_size = 0,
};

PyMODINIT_FUNC PyInit__cext() {
    if (!cuda_loader_init())
        return nullptr;

    // CUDA initialization is lazy - happens on first CUDA operation.
    // This allows CUDA_VISIBLE_DEVICES to be set after importing numba_cuda_mlir.

    PyPtr m = steal(PyModule_Create(&module_def));
    if (!m) return nullptr;

    if (!kernel_init(m.get()))
        return nullptr;

    if (!cuda_helper_init(m.get()))
        return nullptr;

    if (!llvm_downgrade_init(m.get()))
        return nullptr;

    return m.release();
}

