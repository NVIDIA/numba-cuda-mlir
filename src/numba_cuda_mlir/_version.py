# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.resources

__version__ = (
    importlib.resources.files("numba_cuda_mlir").joinpath("VERSION").read_text().strip()
)
