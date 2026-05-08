..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

Memory Management
=================

.. warning:: The Memory API functions are not recommended for use in new code,
   and are provided for backwards compatibility with code written for
   Numba-CUDA. It is recommended that applications pass `PyTorch Tensors
   <https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html>`_
   or `CuPy arrays <https://docs.cupy.dev/en/stable/user_guide/basic.html>`_ to
   kernels instead.

.. autofunction:: numba_cuda_mlir.numba_cuda.to_device
.. autofunction:: numba_cuda_mlir.numba_cuda.device_array
.. autofunction:: numba_cuda_mlir.numba_cuda.device_array_like
.. autofunction:: numba_cuda_mlir.numba_cuda.pinned_array
.. autofunction:: numba_cuda_mlir.numba_cuda.pinned_array_like
.. autofunction:: numba_cuda_mlir.numba_cuda.mapped_array
.. autofunction:: numba_cuda_mlir.numba_cuda.mapped_array_like
.. autofunction:: numba_cuda_mlir.numba_cuda.managed_array
.. autofunction:: numba_cuda_mlir.numba_cuda.pinned
.. autofunction:: numba_cuda_mlir.numba_cuda.mapped

Device Objects
--------------

.. autoclass:: numba_cuda_mlir.numba_cuda.cudadrv.devicearray.DeviceNDArray
   :members: copy_to_device, copy_to_host, is_c_contiguous, is_f_contiguous,
              ravel, reshape, split
.. autoclass:: numba_cuda_mlir.numba_cuda.cudadrv.devicearray.DeviceRecord
   :members: copy_to_device, copy_to_host
.. autoclass:: numba_cuda_mlir.numba_cuda.cudadrv.devicearray.MappedNDArray
   :members: copy_to_device, copy_to_host, split
