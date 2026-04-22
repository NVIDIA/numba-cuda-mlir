# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NRT configuration handling."""

import os

from cusimt.numba_cuda.core import config


def _readenv(name: str, typ: type, default):
    """Read an environment variable and convert to the given type."""
    val = os.environ.get(name)
    if val is None:
        return default
    if typ is bool:
        return val.lower() in ("1", "true", "yes", "on")
    return typ(val)


def is_nrt_enabled() -> bool:
    """Check if NRT is enabled (evaluated at call time, not import time)."""
    return _readenv("NUMBA_CUDA_ENABLE_NRT", bool, False) or getattr(
        config, "NUMBA_CUDA_ENABLE_NRT", False
    )


def is_nrt_stats_enabled() -> bool:
    """Check if NRT statistics are enabled (evaluated at call time)."""
    return _readenv("NUMBA_CUDA_NRT_STATS", bool, False) or getattr(
        config, "NUMBA_CUDA_NRT_STATS", False
    )
