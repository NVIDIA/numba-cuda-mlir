..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _loop-unroll:

Loop Unrolling
==============

If the number of iterations ``n`` of a ``for`` loop is known at compile time, it can be unrolled fully with ``consteval``, which replicates the loop instructions ``n`` times in numba-cuda-mlir's IR before it gets to the NVIDIA compiler.
Finer control can be gained using ``cuda.unroll(iterator, count)``, which leaves the loop intact on the numba-cuda-mlir side but tells the backend how to unroll it.
Using ``cuda.unroll`` lets you specify how many iterations should be unrolled, so a loop of (say) 8 iterations could be unrolled to a 2-iteration loop which has 4 copies of the loop body in the final instruction stream.
The ``libnvvm`` backend has the final say on what loops are unrolled using ``cuda.unroll``. Sometimes, it can unroll loops that you'd prefer to keep rolled, in which case you can use ``cuda.nounroll`` to tell the backend *not* to unroll a loop.

Compile-time unrolling
----------------------

.. function:: numba_cuda_mlir.cuda.experimental.consteval(iterable)

   Evaluate ``iterable`` at compile time and repeat the loop body once per item, with the loop variable replaced by that item. The loop variable may be a name, a tuple or a list. ``break``, ``continue`` and an ``else`` clause are not supported. Enable it with ``from numba_cuda_mlir.cuda.experimental import consteval`` in the module that defines the kernel.

Example::

    from numba_cuda_mlir.cuda.experimental import consteval

    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in consteval(range(16)):
            acc += x[k]
        out[0] = acc


Unroll hints
------------

These functions attach an unroll hint for the backend to a ``for`` loop over a ``range``, a 1-D array, a tuple or ``np.nditer``. These perform the same way as ``#pragma unroll`` does in CUDA C++.

.. seealso:: `Loop unrolling metadata <https://llvm.org/docs/LangRef.html#llvm-loop-unroll>`_ in the LLVM Language Reference.

.. function:: numba_cuda_mlir.cuda.unroll(iterable, count=None)

   Iterate ``iterable`` with a full-unroll hint (``#pragma unroll``), or an unroll-by-``count`` hint (``#pragma unroll count``) when ``count`` is given positionally or by keyword. ``count`` must be a positive compile-time integer, or ``None`` for the full-unroll hint.

.. function:: numba_cuda_mlir.cuda.nounroll(iterable)

   Iterate ``iterable`` with an unroll-disable hint (``#pragma unroll 1``).

Example::

    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(16), count=4):
            acc += x[k]
        out[0] = acc
