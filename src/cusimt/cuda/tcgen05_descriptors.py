# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Common struct and union definitions for tcgen05 (UTCMMA) operations.

This module provides canonical definitions of structs and unions used for
tcgen05 instructions, including shared memory descriptors and instruction
descriptors. These definitions match the hardware specification and can be
reused across different kernels.
"""

import ctypes  # Import at module level for use in device functions
from cusimt.numba_cuda import types
from cusimt.host import struct, union

# ============================================================================
# Instruction Descriptor
# ============================================================================
# Union for tcgen05 instruction descriptor
# Matches the hardware specification for UTCMMA instruction encoding

InstrDescriptorBitfields = struct(
    [
        ("sparse_id2_", types.uint32, 2),
        ("sparse_flag_", types.uint32, 1),
        ("saturate_", types.uint32, 1),
        ("c_format_", types.uint32, 2),
        ("sparse_format_", types.uint32, 1),
        ("a_format_", types.uint32, 3),
        ("b_format_", types.uint32, 3),
        ("a_negate_", types.uint32, 1),
        ("b_negate_", types.uint32, 1),
        ("a_major_", types.uint32, 1),
        ("b_major_", types.uint32, 1),
        ("n_dim_", types.uint32, 6),
        (None, types.uint32, 1),
        ("m_dim_", types.uint32, 5),
        (None, types.uint32, 1),
        ("max_shift_", types.uint32, 2),
    ],
    name="InstrDescriptorBitfields",
)

InstrDescriptor = union(
    [
        ("desc_", types.uint32),
        ("bitfields", InstrDescriptorBitfields),
    ],
    name="InstrDescriptor",
)


# ============================================================================
# Shared Memory Descriptor
# ============================================================================
# Union for tcgen05 shared memory descriptor
# Used to describe shared memory layout for TMA loads

SmemDescriptorBitfields = struct(
    [
        ("start_address_", types.uint64, 14),
        (None, types.uint64, 2),
        ("leading_byte_offset_", types.uint64, 14),
        (None, types.uint64, 2),
        ("stride_byte_offset_", types.uint64, 14),
        ("version_", types.uint64, 2),
        (None, types.uint64, 1),
        ("base_offset_", types.uint64, 3),
        (None, types.uint64, 4),
        (None, types.uint64, 5),
        ("layout_type_", types.uint64, 3),
    ],
    name="SmemDescriptorBitfields",
)

SmemDescriptor = union(
    [
        ("desc_", types.uint64),
        ("reg32_", types.UniTuple(types.uint32, 2)),
        ("reg16_", types.UniTuple(types.uint16, 4)),
        ("bitfields", SmemDescriptorBitfields),
    ],
    name="SmemDescriptor",
)


# ============================================================================
# Device Functions for Building Descriptors
# ============================================================================
# These functions can be called from within JIT-compiled kernels
# They are compiled as device functions using cuda.jit(device=True)


# Import cuda module - use lazy import to avoid circular dependencies
def _get_cuda():
    """Get cuda module, importing lazily to avoid circular imports."""
    import cusimt.cuda as cuda

    return cuda


# Public API functions with default parameters
# These will be compiled as device functions using cuda.jit(device=True)
def tcgen05_build_smem_descriptor(
    smem_ptr,
    leading_byte_offset=16,
    stride_byte_offset=1024,
    base_offset=0,
    layout=2,
):
    """
    Build a tcgen05 shared memory descriptor.

    This function can be called from within JIT-compiled kernels.

    Parameters:
    - smem_ptr: Shared memory pointer (from types.ptr(smem_array))
    - leading_byte_offset: Leading byte offset (default: 16)
    - stride_byte_offset: Stride byte offset (default: 1024)
    - base_offset: Base offset (default: 0)
    - layout: Layout type (default: 2 for SWIZZLE_128B)

    Returns:
    - SmemDescriptor instance with desc_ field ready for use
    """
    smem_desc = SmemDescriptor()
    bitfields = smem_desc.bitfields
    bitfields.start_address_ = ctypes.cast(smem_ptr, ctypes.c_uint32) >> 4
    bitfields.leading_byte_offset_ = leading_byte_offset >> 4
    bitfields.stride_byte_offset_ = stride_byte_offset >> 4
    bitfields.base_offset_ = base_offset
    bitfields.layout_type_ = layout
    smem_desc.bitfields = bitfields
    return smem_desc


def tcgen05_build_idescriptor(
    c_format=1,
    n_dim=16,
    m_dim=8,
):
    """
    Build a tcgen05 instruction descriptor.

    This function can be called from within JIT-compiled kernels.

    Parameters:
    - c_format: C format (default: 1 for F32)
    - n_dim: N dimension (default: 16)
    - m_dim: M dimension (default: 8)

    Returns:
    - InstrDescriptor instance with desc_ field ready for use
    """
    idesc = InstrDescriptor()
    bitfields = idesc.bitfields
    bitfields.c_format_ = c_format
    bitfields.n_dim_ = n_dim
    bitfields.m_dim_ = m_dim
    idesc.bitfields = bitfields
    return idesc


def tcgen05_update_smem_start_address(
    smem_desc,
    offset,
):
    """
    Update the start address of a shared memory descriptor.

    This function can be called from within JIT-compiled kernels.

    Parameters:
    - smem_desc: Existing SmemDescriptor instance
    - offset: Offset to add to the start address (in units of 16 bytes)

    Returns:
    - New SmemDescriptor instance with updated start_address_
    """
    new_desc = SmemDescriptor()
    bitfields = smem_desc.bitfields
    new_bitfields = new_desc.bitfields
    # Copy all fields from the original descriptor
    new_bitfields.start_address_ = bitfields.start_address_ + offset
    new_bitfields.leading_byte_offset_ = bitfields.leading_byte_offset_
    new_bitfields.stride_byte_offset_ = bitfields.stride_byte_offset_
    new_bitfields.version_ = bitfields.version_
    new_bitfields.base_offset_ = bitfields.base_offset_
    new_bitfields.layout_type_ = bitfields.layout_type_
    new_desc.bitfields = new_bitfields
    return new_desc


# Compile as device functions after module is loaded
# This avoids circular import issues by deferring compilation
def _init_device_functions():
    """Initialize device functions by compiling them."""
    cuda = _get_cuda()

    global tcgen05_build_smem_descriptor, tcgen05_build_idescriptor, tcgen05_update_smem_start_address

    # Compile the wrapper functions (with defaults) directly as device functions
    # This matches the pattern used in test examples
    tcgen05_build_smem_descriptor = cuda.jit(device=True, inline="always")(
        tcgen05_build_smem_descriptor
    )
    tcgen05_build_idescriptor = cuda.jit(device=True, inline="always")(
        tcgen05_build_idescriptor
    )
    tcgen05_update_smem_start_address = cuda.jit(device=True, inline="always")(
        tcgen05_update_smem_start_address
    )
