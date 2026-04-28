/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "py.h"
#include "cuda_shim.h"

Status cuda_loader_init();

#define FOREACH_CUDA_FUNCTION_TO_LOAD(X) \
    X(cuInit, 2000) \
    X(cuLibraryLoadData, 12000) \
    X(cuLibraryUnload, 12000) \
    X(cuLibraryGetKernel, 12000) \
    X(cuLibraryGetModule, 12000) \
    X(cuModuleGetGlobal, 3020) \
    X(cuGetErrorString, 6000) \
    X(cuGetErrorName, 6000) \
    X(cuLaunchKernel, 7000) \
    X(cuLaunchKernelEx, 11060) \
    X(cuMemAlloc, 3020) \
    X(cuMemFree, 3020) \
    X(cuMemcpyDtoH, 3020) \
    X(cuMemcpyHtoD, 3020) \
    X(cuPointerGetAttribute, 4000) \
    X(cuCtxPushCurrent, 4000) \
    X(cuCtxPopCurrent, 4000) \
    X(cuCtxGetCurrent, 4000) \
    X(cuCtxSetCurrent, 4000) \
    X(cuCtxSynchronize, 2000) \
    X(cuCtxGetDevice, 2000) \
    X(cuDeviceGet, 2000) \
    X(cuDeviceGetAttribute, 2000) \
    X(cuDevicePrimaryCtxRetain, 7000) \
    X(cuDevicePrimaryCtxRelease, 7000) \
    X(cuKernelGetFunction, 12000) \
    X(cuFuncSetAttribute, 2000) \
    X(cuFuncGetAttribute, 2000)


#define DECLARE_CUDA_FUNC_EXTERN(name, _cuda_version) \
    extern typeof(name)* g_##name;

FOREACH_CUDA_FUNCTION_TO_LOAD(DECLARE_CUDA_FUNC_EXTERN)
