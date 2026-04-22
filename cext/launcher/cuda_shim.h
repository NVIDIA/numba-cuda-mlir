/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Minimal CUDA Driver API shim for build-time independence from the
 * CUDA Toolkit.  Only types, enums, constants, and function declarations
 * actually used by cuSIMT's C extension are declared here.  All CUDA
 * functions are loaded at runtime via dlopen/cuGetProcAddress
 * (see cuda_loader.cpp).
 *
 * Values must match the official cuda.h — they are part of the
 * stable driver ABI.
 */

#ifndef CUSIMT_CUDA_SHIM_H
#define CUSIMT_CUDA_SHIM_H

#include <stdint.h>
#include <stddef.h>

/* ---- basic types ------------------------------------------------------- */

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long cuuint64_t;

/* ---- opaque handles ---------------------------------------------------- */

typedef struct CUctx_st*  CUcontext;
typedef struct CUmod_st*  CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUlib_st*  CUlibrary;
typedef struct CUkern_st* CUkernel;

/* ---- CUresult ---------------------------------------------------------- */

typedef enum {
    CUDA_SUCCESS                = 0,
    CUDA_ERROR_INVALID_VALUE    = 1,
} CUresult;

/* ---- CUdevice_attribute (only values used by cuSIMT) ------------------- */

typedef enum {
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X              = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y              = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z              = 7,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR     = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR     = 76,
} CUdevice_attribute;

/* ---- pointer / function attributes ------------------------------------- */

typedef enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
} CUpointer_attribute;

typedef enum {
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
} CUfunction_attribute;

/* ---- cuGetProcAddress -------------------------------------------------- */

typedef enum {
    CU_GET_PROC_ADDRESS_DEFAULT = 0,
} CUgetProcAddress_flags;

typedef enum {
    CU_GET_PROC_ADDRESS_SUCCESS                 = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND        = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT   = 2,
} CUdriverProcAddressQueryResult;

/* ---- launch attributes ------------------------------------------------- */

typedef enum {
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE                          = 2,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION                    = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
} CUlaunch_attribute;

typedef enum {
    CU_CLUSTER_SCHEDULING_POLICY_DEFAULT        = 0,
    CU_CLUSTER_SCHEDULING_POLICY_SPREAD         = 1,
    CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2,
} CUclusterSchedulingPolicy;

typedef union {
    int cooperative;
    struct { unsigned int x, y, z; } clusterDim;
    int clusterSchedulingPolicyPreference;
    char pad[64];   /* union is 64 bytes in the real header */
} CUlaunchAttributeValue;

typedef struct {
    CUlaunch_attribute id;
    char pad[4];    /* padding between id and value */
    CUlaunchAttributeValue value;
} CUlaunchAttribute;

typedef struct {
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    CUlaunchAttribute *attrs;
    unsigned int numAttrs;
} CUlaunchConfig;

/* ---- function declarations --------------------------------------------- */
/*
 * These provide the signatures needed by typeof() in cuda_loader.h.
 * The functions themselves are never linked — they are loaded at runtime
 * via cuGetProcAddress.
 */

CUresult cuInit(unsigned int Flags);
CUresult cuLibraryLoadData(CUlibrary *library, const void *code,
    void *jitOptions, void *jitOptionsValues, unsigned int numJitOptions,
    void *libraryOptions, void *libraryOptionValues, unsigned int numLibraryOptions);
CUresult cuLibraryUnload(CUlibrary library);
CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library, const char *name);
CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library);
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult cuGetErrorString(CUresult error, const char **pStr);
CUresult cuGetErrorName(CUresult error, const char **pStr);
CUresult cuLaunchKernel(CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra);
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
    void **kernelParams, void **extra);
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext *pctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxSynchronize(void);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel);
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);

#endif /* CUSIMT_CUDA_SHIM_H */
