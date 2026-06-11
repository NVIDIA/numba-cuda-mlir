# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from cuda import core

# core.__version__ returns str like '1.0.1',
# '1.0.2.dev46+g16df11dab', etc.
CUDA_CORE_VERSION = tuple(int(i) for i in core.__version__.split(".") if i.isdigit())
CUDA_CORE_GT_0_6 = CUDA_CORE_VERSION >= (0, 6, 0)
CUDA_CORE_GE_1_0 = CUDA_CORE_VERSION >= (1, 0, 0)
