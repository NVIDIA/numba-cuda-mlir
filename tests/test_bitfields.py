# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from cusimt.numba_cuda import types
import cuda.simt as cs
from test_struct_union_defns import (
    SimpleBitfieldStruct,
    InstrDescriptorSimple,
    TmemAddr,
    SmemDescriptorSimple,
)


def test_simple_bitfield():
    """Test basic bitfield functionality with a simple struct"""

    @cs.jit(dump_mlir=True)
    def test_bitfield_kernel(result: types.Array(types.uint32, 1, "A")):
        bf = SimpleBitfieldStruct()

        # Set bitfields
        bf.flag1 = 1  # Set bit 0
        bf.flag2 = 1  # Set bit 1
        bf.value = 7  # Set bits 2-5 to 0b0111

        # Read back
        f1 = bf.flag1
        f2 = bf.flag2
        v = bf.value

        result[0] = f1
        result[1] = f2
        result[2] = v

    result = cs.to_device(np.zeros(3, dtype=np.uint32))
    test_bitfield_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 1, f"Expected flag1=1, got {result_host[0]}"
    assert result_host[1] == 1, f"Expected flag2=1, got {result_host[1]}"
    assert result_host[2] == 7, f"Expected value=7, got {result_host[2]}"


def test_instr_descriptor():
    """Test InstrDescriptor from minimal_utcmma_1sm.cu (simplified version)"""

    @cs.jit(dump_mlir=True)
    def test_instr_desc_kernel(result: types.Array(types.uint32, 1, "A")):
        desc = InstrDescriptorSimple()

        # Set fields to match the CUDA example
        desc.sparse_flag_ = 0  # bit 2
        desc.c_format_ = 1  # bits 4-5 (F32)
        desc.a_format_ = 0  # bits 7-9 (F16)
        desc.b_format_ = 0  # bits 10-12 (F16)

        # Read back
        result[0] = desc.sparse_flag_
        result[1] = desc.c_format_
        result[2] = desc.a_format_
        result[3] = desc.b_format_

    result = cs.to_device(np.zeros(4, dtype=np.uint32))
    test_instr_desc_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 0
    assert result_host[1] == 1
    assert result_host[2] == 0
    assert result_host[3] == 0


def test_tmem_addr():
    """Test TmemAddr union (already tested, but include for completeness)"""

    @cs.jit(dump_mlir=True)
    def test_tmem_kernel(result: types.Array(types.uint32, 1, "A")):
        addr = TmemAddr()
        addr.raw_addr_ = 0x00050003

        fields = addr.fields
        result[0] = fields.col_id
        result[1] = fields.row_id

    result = cs.to_device(np.zeros(2, dtype=np.uint32))
    test_tmem_kernel[1, 1](result)

    result_host = result.copy_to_host()
    assert result_host[0] == 3
    assert result_host[1] == 5


def test_smem_descriptor_simplified():
    """Test simplified SmemDescriptor with 64-bit bitfields"""

    @cs.jit(dump_mlir=True)
    def test_smem_desc_kernel(result: types.Array(types.uint64, 1, "A")):
        desc = SmemDescriptorSimple()

        # Set fields
        desc.start_address_ = 0x1234  # 14-bit value
        desc.layout_type_ = 2  # 3-bit value (SWIZZLE_128B)

        # Read back
        result[0] = desc.start_address_
        result[1] = desc.layout_type_

    result = cs.to_device(np.zeros(2, dtype=np.uint64))
    test_smem_desc_kernel[1, 1](result)

    result_host = result.copy_to_host()

    assert result_host[0] == 0x1234
    assert result_host[1] == 2


if __name__ == "__main__":
    test_simple_bitfield()
    test_instr_descriptor()
    test_tmem_addr()
    test_smem_descriptor_simplified()
