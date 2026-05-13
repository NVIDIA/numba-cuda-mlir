/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include "cuda_helper.h"
#include "cuda_loader.h"


const char* get_cuda_error(CUresult res) {
    const char* str = nullptr;
    g_cuGetErrorString(res, &str);
    return str ? str : "Unknown error";
}

PyObject* get_max_grid_size(PyObject*, PyObject *args) {
    int device_id;
    if (!PyArg_ParseTuple(args, "i", &device_id))
        return NULL;

    CUdevice dev;
    CUresult res = g_cuDeviceGet(&dev, device_id);
    if (res != CUDA_SUCCESS)
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(res));

    std::array<int, 3> max_grid_size;
    for (int i = 0; i < 3; ++i) {
        res = g_cuDeviceGetAttribute(&max_grid_size[i],
            static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i),
            dev);
        if (res != CUDA_SUCCESS) {
            return PyErr_Format(PyExc_RuntimeError,
                                "cuDeviceGetAttribute: %s", get_cuda_error(res));
        }
    }
    return Py_BuildValue("(iii)", max_grid_size[0], max_grid_size[1], max_grid_size[2]);
}

PyObject* get_compute_capability(PyObject*, PyObject*) {
    int major, minor;
    CUdevice dev;
    CUresult res = g_cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(res));
    }
    res = g_cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGetAttribute: %s", get_cuda_error(res));
    }
    res = g_cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGetAttribute: %s", get_cuda_error(res));
    }
    return Py_BuildValue("(ii)", major, minor);
}

static PyMethodDef functions[] = {
    {"get_compute_capability", get_compute_capability, METH_NOARGS,
        "Get compute capability of the default CUDA device"},
    {"_get_max_grid_size", get_max_grid_size, METH_VARARGS,
        "Get max grid size of a CUDA device, given device id"},
    NULL
};

Status cuda_helper_init(PyObject* m) {
    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    return OK;
}
