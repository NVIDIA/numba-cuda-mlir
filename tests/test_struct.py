# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir import cuda
from test_struct_union_defns import (
    SimpleStruct,
    Point,
    InnerStruct,
    OuterStruct,
    BitfieldStructWithPadding,
    PaddedBitfieldStruct,
)


def test_struct_definition():
    """Test that we can define a struct type"""
    assert SimpleStruct.name == "SimpleStruct"
    assert len(SimpleStruct.fields) == 2
    assert SimpleStruct.fields[0] == ("a", types.int32, None)
    assert SimpleStruct.fields[1] == ("b", types.float32, None)


def test_struct_host_construction():
    """Test that we can create struct instances on the host"""
    # Create an instance
    p = Point(x=10, y=20)

    # Test field access
    assert p.x == 10
    assert p.y == 20

    # Test repr
    assert "Point" in repr(p)
    assert "10" in repr(p)
    assert "20" in repr(p)


def test_nested_struct_host_construction():
    """Test that we can create nested structs on the host"""
    # Create nested struct instance
    inner = InnerStruct(a=10, b=20)
    outer = OuterStruct(x=1, inner=inner, y=2)

    # Test field access
    assert outer.x == 1
    assert outer.y == 2
    assert outer.inner.a == 10
    assert outer.inner.b == 20


def test_device_side_struct_construction():
    @cuda.jit(dump_mlir=True)
    def device_struct_kernel(result: types.Array(types.int32, 1, "A")) -> types.void:
        # Construct struct on the device stack using struct literal
        # This will allocate the struct as a local variable and build it field-by-field
        p = Point()  # Allocate uninitialized struct
        p.x = 10  # Set first field
        p.y = 20  # Set second field

        # Use the struct fields
        result[0] = p.x + p.y
        result[1] = p.x
        result[2] = p.y

    result = cuda.to_device(np.zeros(3, dtype=np.int32))
    device_struct_kernel[1, 1](result)

    result_host = result.copy_to_host()
    assert result_host[0] == 30, f"Expected sum=30, got {result_host[0]}"
    assert result_host[1] == 10, f"Expected x=10, got {result_host[1]}"
    assert result_host[2] == 20, f"Expected y=20, got {result_host[2]}"


def test_bitfield_with_padding():
    """Test bitfields with explicit padding"""

    @cuda.jit
    def bitfield_padding_kernel(
        result: types.Array(types.uint32, 1, "A"),
    ) -> types.void:
        s = BitfieldStructWithPadding()

        # Set values
        s.flag = 1  # bit 0
        s.value = 15  # bits 4-7
        s.code = 0xABCD  # bits 16-31

        # Read back
        result[0] = s.flag
        result[1] = s.value
        result[2] = s.code

    result = cuda.to_device(np.zeros(3, dtype=np.uint32))
    bitfield_padding_kernel[1, 1](result)

    result_host = result.copy_to_host()
    assert result_host[0] == 1, f"Expected flag=1, got {result_host[0]}"
    assert result_host[1] == 15, f"Expected value=15, got {result_host[1]}"
    assert result_host[2] == 0xABCD, f"Expected code=0xABCD, got 0x{result_host[2]:04x}"


def test_bitfield_padding_host():
    """Test that padding works correctly on host side"""
    # Create instance on host
    s = PaddedBitfieldStruct(a=5, b=200)

    # Check accessible fields
    assert s.a == 5
    assert s.b == 200

    # Check that padding field is not accessible
    assert "a" in s._fields
    assert "b" in s._fields
    # No None key should exist in _fields
    assert None not in s._fields

    # Repr should not show padding
    repr_str = repr(s)
    assert "a=5" in repr_str
    assert "b=200" in repr_str


if __name__ == "__main__":
    test_struct_definition()
    test_struct_host_construction()
    test_nested_struct_host_construction()
    test_device_side_struct_construction()
    test_bitfield_with_padding()
    test_bitfield_padding_host()
