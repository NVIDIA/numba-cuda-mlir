..
   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _numba-cuda-installation:

============
Installation
============

Requirements
============

Supported GPUs
--------------

Numba CUDA MLIR supports NVIDIA GPUs from Compute Capability 7.0 (Volta). When
used with CUDA Toolkit 12, the support range includes Compute Capabilities from
7.0 to 12.1 depending on the exact installed version, and for CUDA 13 it ranges
from 7.5 to 12.1 (the latest as of CUDA 13.2).


Supported CUDA Toolkits
-----------------------

Numba CUDA MLIR aims to support all minor versions of the two most recent CUDA
Toolkit releases. Presently 12 and 13 are supported.

For further information about version compatibility between toolkit and driver
versions, refer to :ref:`minor-version-compatibility`.


Installation with pip
=====================

Install CUDA 12 dependencies from PyPI via ``pip``::

    $ pip install numba-cuda-mlir[cu12]

CUDA 13 dependencies can be installed with::

    $ pip install numba-cuda-mlir[cu13]
