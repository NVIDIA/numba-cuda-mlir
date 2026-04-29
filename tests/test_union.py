# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from cusimt.numba_cuda import types
import cuda.simt as cuda
from test_struct_union_defns import (
    Uint32FloatUnion,
    TmemAddr,
    PaddedBitfieldFields,
    PaddedBitfieldUnion,
    PaddedBitfieldStruct8,
)


def test_union_definition():
    """Test that we can define a union type"""
    assert Uint32FloatUnion.name == "Uint32FloatUnion"
    assert len(Uint32FloatUnion.variants) == 2
    assert Uint32FloatUnion.variants[0] == ("as_int", types.uint32)
    assert Uint32FloatUnion.variants[1] == ("as_float", types.float32)


def test_union_host_construction():
    """Test that we can create union instances on the host"""
    # Create an instance
    u = Uint32FloatUnion(as_int=42)

    # Test variant access
    assert u.as_int == 42

    # Test repr
    assert "Uint32FloatUnion" in repr(u)


def test_tmem_addr_union():
    @cuda.jit
    def test_tmem_addr_kernel(result: types.Array(types.uint32, 1, "A")) -> types.void:
        # Create TmemAddr on device
        addr = TmemAddr()

        # Set raw_addr_ to a test value
        # col_id should be lower 16 bits, row_id should be upper 16 bits
        addr.raw_addr_ = 0x00050003  # row_id=5, col_id=3

        # Read back via fields
        fields = addr.fields
        col = fields.col_id
        row = fields.row_id

        # Store results
        result[0] = col  # Should be 3
        result[1] = row  # Should be 5
        result[2] = addr.raw_addr_  # Should be 0x00050003

    result = cuda.to_device(np.zeros(3, dtype=np.uint32))
    test_tmem_addr_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 3, f"Expected col_id=3, got {result_host[0]}"
    assert result_host[1] == 5, f"Expected row_id=5, got {result_host[1]}"
    assert result_host[2] == 0x00050003, f"Expected raw_addr=0x00050003, got 0x{result_host[2]:08x}"


def test_union_with_padded_bitfield_struct():
    """Test union containing a struct with padding in bitfields"""

    @cuda.jit(dump_mlir=True)
    def test_padded_union_kernel(
        result: types.Array(types.uint32, 1, "A"),
    ) -> types.void:
        u = PaddedBitfieldUnion()

        # Set via raw
        u.raw = 0xABCD00EF  # y=0xABCD (bits 16-31), padding=0x00, x=0xEF (bits 0-7)

        # Read via fields
        fields = u.fields
        x = fields.x
        y = fields.y

        # Store results
        result[0] = x  # Should be 0xEF
        result[1] = y  # Should be 0xABCD
        result[2] = u.raw  # Should be 0xABCD00EF

    result = cuda.to_device(np.zeros(3, dtype=np.uint32))
    test_padded_union_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 0xEF, f"Expected x=0xEF, got 0x{result_host[0]:02x}"
    assert result_host[1] == 0xABCD, f"Expected y=0xABCD, got 0x{result_host[1]:04x}"
    assert result_host[2] == 0xABCD00EF, f"Expected raw=0xABCD00EF, got 0x{result_host[2]:08x}"


def test_padded_struct_host():
    """Test that padding works correctly in structs on host side"""
    # Create instance
    s = PaddedBitfieldStruct8(a=7, b=15)

    # Check values
    assert s.a == 7
    assert s.b == 15

    # Repr should not show padding
    repr_str = repr(s)
    assert "a=7" in repr_str
    assert "b=15" in repr_str
    assert "None" not in repr_str


if __name__ == "__main__":
    test_union_definition()
    test_union_host_construction()
    test_tmem_addr_union()
    test_union_with_padded_bitfield_struct()
    test_padded_struct_host()
