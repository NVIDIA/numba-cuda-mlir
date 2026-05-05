# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir._version import __version__
from numba_cuda_mlir.mlir import make_nanobind_metaclass_inheritable

make_nanobind_metaclass_inheritable()

__all__ = ["cuda", "__version__"]
