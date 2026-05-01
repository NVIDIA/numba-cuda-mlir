# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache


@lru_cache
def setup_lowering_patches():
    """Apply necessary patches for lowering system."""
    import numba_cuda_mlir.numba_cuda.misc.literal  # noqa: F401


@lru_cache
def register_lowering():
    """Deprecated: Use MLIRTargetContext.load_additional_registries() instead."""
    setup_lowering_patches()
