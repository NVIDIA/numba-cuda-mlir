..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

========================
IPC Support (Deprecated)
========================

.. warning:: IPC support is deprecated and only provided for backwards
   compatibility with code written for Numba-CUDA. Users are encouraged to use
   the IPC facilities of another library that provides CUDA arrays, such as
   CuPy, PyTorch, etc.

.. _cuda-ipc-memory:

Sharing between process
=======================

Sharing between processes is implemented using the Legacy CUDA IPC API
(functions whose names begin with ``cuIpc``), and is supported only on Linux.


Export device array to another process
--------------------------------------

A device array can be shared with another process in the same machine using
the CUDA IPC API.  To do so, use the ``.get_ipc_handle()`` method on the device
array to get a ``IpcArrayHandle`` object, which can be transferred to another
process.


.. automethod:: numba_cuda_mlir.numba_cuda.cudadrv.devicearray.DeviceNDArray.get_ipc_handle
    :noindex:

.. autoclass:: numba_cuda_mlir.numba_cuda.cudadrv.devicearray.IpcArrayHandle
    :members: open, close


Import IPC memory from another process
--------------------------------------

The following function is used to open IPC handle from another process
as a device array.

.. autofunction:: numba_cuda_mlir.cuda.open_ipc_array
