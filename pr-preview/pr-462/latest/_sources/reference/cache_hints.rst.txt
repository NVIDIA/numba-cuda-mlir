..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _cache-hints:

Cache Hints for Memory Operations
=================================

These functions provide explicit control over caching behavior for memory
operations. They generate PTX instructions with cache policy hints that can
optimize specific memory access patterns. All functions support arrays or
pointers with all bitwidths of signed/unsigned integer and floating-point
types.

.. seealso:: `Cache Operators
   <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_
   in the PTX ISA documentation.

.. function:: numba_cuda_mlir.cuda.ldca(array, i)

   Load element ``i`` from ``array`` with cache-all policy (``ld.global.ca``). This
   is the default caching behavior.

.. function:: numba_cuda_mlir.cuda.ldcg(array, i)

   Load element ``i`` from ``array`` with cache-global policy (``ld.global.cg``).
   Useful for data shared across thread blocks.

.. function:: numba_cuda_mlir.cuda.ldcs(array, i)

   Load element ``i`` from ``array`` with cache-streaming policy
   (``ld.global.cs``). Optimized for streaming data accessed once.

.. function:: numba_cuda_mlir.cuda.ldlu(array, i)

   Load element ``i`` from ``array`` with last-use policy (``ld.global.lu``).
   Indicates data is unlikely to be reused.

.. function:: numba_cuda_mlir.cuda.ldcv(array, i)

   Load element ``i`` from ``array`` with cache-volatile policy (``ld.global.cv``).
   Used for volatile data that may change externally.

.. function:: numba_cuda_mlir.cuda.stcg(array, i, value)

   Store ``value`` to ``array[i]`` with cache-global policy (``st.global.cg``).
   Useful for data shared across thread blocks.

.. function:: numba_cuda_mlir.cuda.stcs(array, i, value)

   Store ``value`` to ``array[i]`` with cache-streaming policy (``st.global.cs``).
   Optimized for streaming writes.

.. function:: numba_cuda_mlir.cuda.stwb(array, i, value)

   Store ``value`` to ``array[i]`` with write-back policy (``st.global.wb``). This
   is the default caching behavior.

.. function:: numba_cuda_mlir.cuda.stwt(array, i, value)

   Store ``value`` to ``array[i]`` with write-through policy (``st.global.wt``).
   Writes through cache hierarchy to memory.
