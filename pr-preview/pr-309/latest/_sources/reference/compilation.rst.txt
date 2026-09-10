..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

Compilation API
===============

Numba CUDA MLIR provides an entry point for compiling a Python function without
invoking any of the driver API. This can be useful for:

- Generating PTX that is to be inlined into other PTX code (e.g. from outside
  the Numba CUDA MLIR / Python ecosystem).
- Generating PTX or LTO-IR to link with objects from non-Python translation
  units.
- Generating code when there is no device present.
- Generating code prior to a fork without initializing CUDA.

.. note:: It is the user's responsibility to manage any ABI issues arising from
   the use of compilation to PTX / LTO-IR. Passing the ``abi="c"`` keyword
   argument can provide a solution to most issues that may arise - see
   :ref:`cuda-using-the-c-abi`.

.. autofunction:: numba_cuda_mlir.cuda.compile

.. autofunction:: numba_cuda_mlir.cuda.compile_all


The environment variable ``NUMBA_CUDA_DEFAULT_PTX_CC`` can be set to control
the default compute capability targeted by ``compile`` - see
:ref:`numba-envvars-gpu-support`. If code for the compute capability of the
current device is required, the ``compile_for_current_device`` function can
be used:

.. autofunction:: numba_cuda_mlir.cuda.compile_for_current_device


Numba CUDA MLIR also provides two functions that may be used in legacy code
that specifically compile to PTX only:

.. autofunction:: numba_cuda_mlir.cuda.compile_ptx

.. autofunction:: numba_cuda_mlir.cuda.compile_ptx_for_current_device
