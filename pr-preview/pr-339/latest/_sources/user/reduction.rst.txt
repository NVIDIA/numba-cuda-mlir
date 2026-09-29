..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

GPU Reduction
==============

.. warning:: The reduction decorator is deprecated and provided for backward
   compatibility with code written for Numba-CUDA. Users are recommended to use
   the `cuda.compute parallel computing primitives
   <https://nvidia.github.io/cccl/unstable/python/compute/index.html>`_ from the
   CUDA Core Compute Libraries for new code.

Writing a reduction algorithm for CUDA GPU can be tricky. Numba CUDA MLIR
provides a ``@reduce`` decorator for converting a simple binary operation into
a reduction kernel. An example follows::

    import numpy
    from numba_cuda_mlir import cuda

    @cuda.reduce
    def sum_reduce(a, b):
        return a + b

    A = (numpy.arange(1234, dtype=numpy.float64)) + 1
    expect = A.sum()      # NumPy sum reduction
    got = sum_reduce(A)   # cuda sum reduction
    assert expect == got

Lambda functions can also be used here::

    sum_reduce = cuda.reduce(lambda a, b: a + b)

The Reduce class
----------------

The ``reduce`` decorator creates an instance of the ``Reduce`` class.
Currently, ``reduce`` is an alias to ``Reduce``, but this behavior is not
guaranteed.

.. autoclass:: numba_cuda_mlir.cuda.Reduce
   :members: __init__, __call__
   :member-order: bysource
