..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause


.. _cuda-fast-math:

CUDA Fast Math
==============

As noted in :ref:`fast-math`, for certain classes of applications that utilize
floating point, strict IEEE-754 conformance is not required. For this subset of
applications, performance speedups may be possible.

The CUDA target implements :ref:`fast-math` behavior with two differences.

* First, ``fastmath=True`` enables every flag except ``nnan`` and ``ninf``, and
  there is an extra ``ftz`` flag.

* Secondly, with ``afn`` set, calls to a subset of math module functions on
  ``float32`` operands will be implemented using fast approximate
  implementations from the libdevice library.

  - :func:`math.cos`: Implemented using `__nv_fast_cosf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_cosf.html>`_.
  - :func:`math.sin`: Implemented using `__nv_fast_sinf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sinf.html>`_.
  - :func:`math.tan`: Implemented using `__nv_fast_tanf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_tanf.html>`_.
  - :func:`math.exp`: Implemented using `__nv_fast_expf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_expf.html>`_.
  - :func:`math.log2`: Implemented using `__nv_fast_log2f <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log2f.html>`_.
  - :func:`math.log10`: Implemented using `__nv_fast_log10f <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log10f.html>`_.
  - :func:`math.log`: Implemented using `__nv_fast_logf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_logf.html>`_.
  - :func:`math.pow`: Implemented using `__nv_fast_powf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_powf.html>`_.

The available flags are:

- ``ftz``: flushing of denormals to zero.
- ``afn``: use of a fast approximation to the square root function, of
  ``tanh.approx.f32`` for ``float32`` :func:`math.tanh` on sm_75 and later, and
  of the libdevice functions above.
- ``arcp``: use of a fast approximation to the division operation.
- ``contract``: contraction of multiply and add operations into single fused
  multiply-add operations.
- ``nnan``, ``ninf``, ``nsz`` and ``reassoc``: the LLVM fast-math flags of the
  same names on floating-point operations.
- ``fast``: every flag above except ``nnan`` and ``ninf``; the same as
  ``fastmath=True``.

See the `documentation for nvvmCompileProgram <https://docs.nvidia.com/cuda/libnvvm-api/group__compilation.html#group__compilation_1g76ac1e23f5d0e2240e78be0e63450346>`_ for more details of the ``ftz``, ``afn``, ``arcp`` and ``contract`` optimizations.
