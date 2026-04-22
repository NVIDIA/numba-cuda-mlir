# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache


def _patch_iter_loop_canonicalization():
    """Patch IterLoopCanonicalization to accept both literal_unroll versions."""
    from numba.core.untyped_passes import IterLoopCanonicalization
    from numba.misc.special import literal_unroll as core_literal_unroll
    from cusimt.numba_cuda.misc.special import literal_unroll as cuda_literal_unroll

    # Add cuda's literal_unroll to accepted calls if not already present
    if cuda_literal_unroll not in IterLoopCanonicalization._accepted_calls:
        IterLoopCanonicalization._accepted_calls = (
            core_literal_unroll,
            cuda_literal_unroll,
        )


@lru_cache
def setup_lowering_patches():
    """Apply necessary patches for lowering system."""
    _patch_iter_loop_canonicalization()
    import numba.misc.literal  # noqa: F401


@lru_cache
def register_lowering():
    """Deprecated: Use MLIRTargetContext.load_additional_registries() instead."""
    setup_lowering_patches()
