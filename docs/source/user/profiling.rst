..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _cuda-profiling:

Profiling
=========

The NVidia Visual Profiler can be used directly on executing CUDA Python code -
it is not a requirement to insert calls to these functions into user code.
However, these functions can be used to allow profiling to be performed
selectively on specific portions of the code. For further information on
profiling, see the `NVidia Profiler User's Guide
<https://docs.nvidia.com/cuda/profiler-users-guide/>`_.

.. autofunction:: numba.cuda.profile_start
.. autofunction:: numba.cuda.profile_stop
.. autofunction:: numba.cuda.profiling
