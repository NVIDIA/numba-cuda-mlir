# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-management test configuration."""

import os

import pytest
from cusimt.numba_cuda.core import config as cuda_config


@pytest.fixture(scope="session", autouse=True)
def enable_nrt_for_memory_tests():
    """Enable NRT for all memory management tests."""
    os.environ["NUMBA_CUDA_ENABLE_NRT"] = "1"
    cuda_config.NUMBA_CUDA_ENABLE_NRT = True
    cuda_config.CUDA_ENABLE_NRT = True
    yield
