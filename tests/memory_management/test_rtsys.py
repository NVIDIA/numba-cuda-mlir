# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NRT runtime system (rtsys)."""

import pytest
from numba_cuda_mlir import cuda


@pytest.fixture(autouse=True)
def reset_rtsys():
    """Reset rtsys before and after each test."""
    from numba_cuda_mlir.memory_management import rtsys

    rtsys._reset()
    yield
    rtsys._reset()


def test_rtsys_allocate_initialize():
    """Test that rtsys can allocate and initialize memsys."""
    from numba_cuda_mlir.memory_management import rtsys

    # Initially not allocated or initialized
    assert rtsys._memsys is None
    assert rtsys._memsys_library is None
    assert rtsys._initialized is False

    # ensure_initialized should allocate and initialize
    rtsys.ensure_initialized()

    assert rtsys._memsys is not None
    assert rtsys._memsys_library is not None
    assert rtsys._initialized is True


def test_rtsys_stats_enable_disable():
    """Test that stats can be toggled on and off."""
    from numba_cuda_mlir.memory_management import rtsys

    rtsys.ensure_initialized()

    # Initially stats are disabled (unless NUMBA_CUDA_NRT_STATS is set)
    # Enable stats
    rtsys.memsys_enable_stats()
    assert rtsys.memsys_stats_enabled() is True

    # Disable stats
    rtsys.memsys_disable_stats()
    assert rtsys.memsys_stats_enabled() is False

    # Enable again
    rtsys.memsys_enable_stats()
    assert rtsys.memsys_stats_enabled() is True


def test_rtsys_stats_query():
    """Test that stats can be queried when enabled."""
    from numba_cuda_mlir.memory_management import rtsys, _nrt_mstats

    rtsys.ensure_initialized()
    rtsys.memsys_enable_stats()

    # Get allocation stats
    stats = rtsys.get_allocation_stats()

    assert isinstance(stats, _nrt_mstats)
    # Initially all stats should be 0
    assert stats.alloc == 0
    assert stats.free == 0
    assert stats.mi_alloc == 0
    assert stats.mi_free == 0


def test_rtsys_stats_query_individual():
    """Test that individual stats can be queried."""
    from numba_cuda_mlir.memory_management import rtsys

    rtsys.ensure_initialized()
    rtsys.memsys_enable_stats()

    # Query individual stats
    assert rtsys.memsys_get_stats_alloc() == 0
    assert rtsys.memsys_get_stats_free() == 0
    assert rtsys.memsys_get_stats_mi_alloc() == 0
    assert rtsys.memsys_get_stats_mi_free() == 0


def test_rtsys_stats_query_raises_when_disabled():
    """Test that querying stats raises when stats are disabled."""
    from numba_cuda_mlir.memory_management import rtsys

    rtsys.ensure_initialized()
    rtsys.memsys_disable_stats()

    with pytest.raises(RuntimeError, match="NRT stats are disabled"):
        rtsys.get_allocation_stats()

    with pytest.raises(RuntimeError, match="NRT stats are disabled"):
        rtsys.memsys_get_stats_alloc()


def test_rtsys_idempotent():
    """Test that ensure_initialized is idempotent."""
    from numba_cuda_mlir.memory_management import rtsys

    rtsys.ensure_initialized()
    memsys1 = rtsys._memsys
    library1 = rtsys._memsys_library

    # Call again - should be no-op
    rtsys.ensure_initialized()

    assert rtsys._memsys is memsys1
    assert rtsys._memsys_library is library1
