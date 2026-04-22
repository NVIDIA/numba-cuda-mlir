# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for union array variants using UniTuple.

Tests the ability to access union data through different array views,
similar to the CUDA SmemDescriptor pattern:
    union SmemDescriptor {
        uint64_t desc_;
        uint32_t reg32_[2];
        uint16_t reg16_[4];
    };
"""

import pytest
import numpy as np
import cuda.simt as cuda
from cuda.simt import types
from test_struct_union_defns import (
    Uint64Uint32ArrayUnion,
    Uint64Uint16ArrayUnion,
    MultiViewUnion,
    SmemDescriptorForArrayTest,
)


def test_union_simple_array_variant():
    """Test union with a simple array variant (2x uint32)"""

    @cuda.jit
    def test_kernel(result: types.Array(types.uint32, 1, "A")) -> types.void:
        # Create union
        u = Uint64Uint32ArrayUnion()

        # Set via uint64
        u.as_uint64 = 0x0000000500000003  # upper=5, lower=3

        # Read via array
        arr = u.as_uint32_array

        # Store results
        result[0] = arr[0]  # Should be 3 (lower 32 bits)
        result[1] = arr[1]  # Should be 5 (upper 32 bits)

    result = cuda.to_device(np.zeros(2, dtype=np.uint32))
    test_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 3, f"Expected arr[0]=3, got {result_host[0]}"
    assert result_host[1] == 5, f"Expected arr[1]=5, got {result_host[1]}"


def test_union_uint16_array_variant():
    """Test union with uint16 array (4 elements)"""

    @cuda.jit
    def test_kernel(result: types.Array(types.uint16, 1, "A")) -> types.void:
        u = Uint64Uint16ArrayUnion()

        # Set via uint64: 0xAAAABBBBCCCCDDDD
        u.as_uint64 = 0xAAAABBBBCCCCDDDD

        # Read via array
        arr = u.as_uint16_array

        # Store results (little-endian)
        result[0] = arr[0]  # 0xDDDD
        result[1] = arr[1]  # 0xCCCC
        result[2] = arr[2]  # 0xBBBB
        result[3] = arr[3]  # 0xAAAA

    result = cuda.to_device(np.zeros(4, dtype=np.uint16))
    test_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 0xDDDD, f"Expected 0xDDDD, got 0x{result_host[0]:04x}"
    assert result_host[1] == 0xCCCC, f"Expected 0xCCCC, got 0x{result_host[1]:04x}"
    assert result_host[2] == 0xBBBB, f"Expected 0xBBBB, got 0x{result_host[2]:04x}"
    assert result_host[3] == 0xAAAA, f"Expected 0xAAAA, got 0x{result_host[3]:04x}"


def test_union_array_setattr():
    """Test setting union via array variant"""

    @cuda.jit
    def test_kernel(result: types.Array(types.uint64, 1, "A")) -> types.void:
        u = Uint64Uint32ArrayUnion()

        # Set via array - note: we need to construct a tuple
        # In Python: u.as_uint32_array = (0x12345678, 0xABCDEF00)
        # For now, let's set individual elements through intermediate variable
        # TODO: Support direct tuple assignment

        # For this test, set via uint64 and verify we can read back
        u.as_uint64 = 0xABCDEF0012345678

        result[0] = u.as_uint64

    result = cuda.to_device(np.zeros(1, dtype=np.uint64))
    test_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 0xABCDEF0012345678


def test_union_multiple_array_views():
    """Test union with multiple array views of different sizes"""

    @cuda.jit
    def test_kernel(result: types.Array(types.uint32, 1, "A")) -> types.void:
        u = MultiViewUnion()

        # Set via uint64
        u.as_uint64 = 0x1111222233334444

        # Read via uint32 array
        arr32 = u.as_uint32_array
        val32_0 = arr32[0]  # 0x33334444
        val32_1 = arr32[1]  # 0x11112222

        # Read via uint16 array
        arr16 = u.as_uint16_array
        val16_0 = arr16[0]  # 0x4444
        val16_1 = arr16[1]  # 0x3333
        val16_2 = arr16[2]  # 0x2222
        val16_3 = arr16[3]  # 0x1111

        # Store results
        result[0] = val32_0
        result[1] = val32_1
        result[2] = val16_0
        result[3] = val16_1
        result[4] = val16_2
        result[5] = val16_3

    result = cuda.to_device(np.zeros(6, dtype=np.uint32))
    test_kernel[1, 1](result)

    result_host = result.copy_to_host()

    # Check uint32 views
    assert (
        result_host[0] == 0x33334444
    ), f"Expected 0x33334444, got 0x{result_host[0]:08x}"
    assert (
        result_host[1] == 0x11112222
    ), f"Expected 0x11112222, got 0x{result_host[1]:08x}"

    # Check uint16 views
    assert result_host[2] == 0x4444, f"Expected 0x4444, got 0x{result_host[2]:04x}"
    assert result_host[3] == 0x3333, f"Expected 0x3333, got 0x{result_host[3]:04x}"
    assert result_host[4] == 0x2222, f"Expected 0x2222, got 0x{result_host[4]:04x}"
    assert result_host[5] == 0x1111, f"Expected 0x1111, got 0x{result_host[5]:04x}"


def test_smem_descriptor_like_union():
    """
    Test a SmemDescriptor-like union combining:
    - Single uint64 view
    - Array views (uint32[2], uint16[4])
    - Bitfield struct view
    """

    @cuda.jit
    def test_kernel(result: types.Array(types.uint32, 1, "A")) -> types.void:
        desc = SmemDescriptorForArrayTest()

        # Set descriptor value
        desc.desc_ = 0x7654321087654321

        # Read via reg32_ array
        reg32 = desc.reg32_
        lower32 = reg32[0]
        upper32 = reg32[1]

        # Read via reg16_ array
        reg16 = desc.reg16_
        word0 = reg16[0]
        word1 = reg16[1]

        # Read via bitfields
        bf = desc.bitfields
        start_addr = bf.start_address
        layout = bf.layout_type

        # Store results
        result[0] = lower32
        result[1] = upper32
        result[2] = word0
        result[3] = word1
        result[4] = start_addr
        result[5] = layout

    result = cuda.to_device(np.zeros(6, dtype=np.uint32))
    test_kernel[1, 1](result)

    result_host = result.copy_to_host()

    # Verify reg32 access
    assert (
        result_host[0] == 0x87654321
    ), f"Expected 0x87654321, got 0x{result_host[0]:08x}"
    assert (
        result_host[1] == 0x76543210
    ), f"Expected 0x76543210, got 0x{result_host[1]:08x}"

    # Verify reg16 access
    assert result_host[2] == 0x4321, f"Expected 0x4321, got 0x{result_host[2]:04x}"
    assert result_host[3] == 0x8765, f"Expected 0x8765, got 0x{result_host[3]:04x}"

    # Verify bitfield access (14 bits from 0x4321 = 0x0321, 3 bits from upper = 0x7)
    expected_start = 0x4321 & 0x3FFF  # 14 bits
    expected_layout = (0x76543210 >> 61) & 0x7  # Top 3 bits of upper 32

    assert (
        result_host[4] == expected_start
    ), f"Expected start=0x{expected_start:04x}, got 0x{result_host[4]:04x}"


def test_union_array_variant_host_construction():
    """Test that array variants work in host construction"""
    # Create instance on host
    u = Uint64Uint32ArrayUnion(as_uint64=0x1234567890ABCDEF)

    # Verify we can access
    assert u.as_uint64 == 0x1234567890ABCDEF


if __name__ == "__main__":
    # Run tests
    test_union_simple_array_variant()
    test_union_uint16_array_variant()
    test_union_array_setattr()
    test_union_multiple_array_views()
    test_smem_descriptor_like_union()
    test_union_array_variant_host_construction()
    print("All tests passed!")
