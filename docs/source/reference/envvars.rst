..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _numba-envvars-gpu-support:

Environment Variables
---------------------

Various environment variables can be set to affect the behavior of the CUDA
target.

.. envvar:: NUMBA_CUDA_ARRAY_INTERFACE_SYNC

   Whether to synchronize on streams provided by objects imported using the CUDA
   Array Interface. This defaults to 1. If set to 0, then no synchronization
   takes place, and the user of Numba CUDA MLIR (and other CUDA libraries) is
   responsible for ensuring correctness with respect to synchronization on
   streams.

.. envvar:: NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS

   Enable warnings if the grid size is too small relative to the number of
   streaming multiprocessors (SM). This option is on by default (default value is 1).

   The heuristic checked is whether ``gridsize < 2 * (number of SMs)``. NOTE: The absence of
   a warning does not imply a good gridsize relative to the number of SMs. Disabling
   this warning will reduce the number of CUDA API calls (during JIT compilation), as the
   heuristic needs to check the number of SMs available on the device in the
   current context.

.. envvar:: NUMBA_CUDA_WARN_ON_IMPLICIT_COPY

   Enable warnings if a kernel is launched with host memory which forces a copy to and
   from the device. This option is on by default (default value is 1).

.. envvar:: NUMBA_CUDA_INCLUDE_PATH

   The location of the CUDA include files. This is used when linking CUDA C++
   sources to Python kernels, and needs to be correctly set for CUDA includes to
   be available to linked C/C++ sources. On Linux, it defaults to
   ``/usr/local/cuda/include``. On Windows, the default is
   ``$env:CUDA_PATH\include``.

.. envvar:: NUMBA_CUDA_NVRTC_EXTRA_SEARCH_PATHS

   A colon separated list of paths that NVRTC should search for when compiling
   external functions. These folders are searched after the system cudatoolkit
   search paths and Numba CUDA MLIR's internal search paths.
