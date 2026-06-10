# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
from cuda import core

# Utilize core._version attributes
#    __version__ => '1.0.2.dev46+g16df11dab'
#    __version_tuple__ => (1, 0, 2, 'dev46', 'g16df11dab')
CUDA_CORE_VERSION = core._version.__version_tuple__
CUDA_CORE_GT_0_6 = CUDA_CORE_VERSION >= (0, 6, 0)
CUDA_CORE_GE_1_0 = CUDA_CORE_VERSION >= (1, 0, 0)
