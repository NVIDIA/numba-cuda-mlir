# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Common struct and union definitions used across test files.

This module provides canonical definitions of structs and unions to avoid
duplication across test files. Each definition is named descriptively to
indicate its purpose and structure.
"""

from numba import types
from cuda.simt import host

struct = host.struct
union = host.union


# ============================================================================
# Simple Test Structs
# ============================================================================

SimpleStruct = struct(
    [
        ("a", types.int32),
        ("b", types.float32),
    ],
    name="SimpleStruct",
)

Point = struct(
    [
        ("x", types.int32),
        ("y", types.int32),
    ],
    name="Point",
)

InnerStruct = struct(
    [
        ("a", types.int32),
        ("b", types.int32),
    ],
)

OuterStruct = struct(
    [
        ("x", types.int32),
        ("inner", InnerStruct._type),
        ("y", types.int32),
    ],
)


# ============================================================================
# Simple Bitfield Structs
# ============================================================================

SimpleBitfieldStruct = struct(
    [
        ("flag1", types.uint32, 1),  # 1-bit field at offset 0
        ("flag2", types.uint32, 1),  # 1-bit field at offset 1
        ("value", types.uint32, 4),  # 4-bit field at offset 2
    ],
    name="SimpleBitfieldStruct",
)

BitfieldStructWithPadding = struct(
    [
        ("flag", types.uint32, 1),  # 1 bit
        (None, types.uint32, 3),  # 3 bits padding
        ("value", types.uint32, 4),  # 4 bits
        (None, types.uint32, 8),  # 8 bits padding
        ("code", types.uint32, 16),  # 16 bits
    ],
)

PaddedBitfieldStruct = struct(
    [
        ("a", types.uint16, 4),  # 4 bits
        (None, types.uint16, 4),  # 4 bits padding
        ("b", types.uint16, 8),  # 8 bits
    ],
)

PaddedBitfieldStruct8 = struct(
    [
        ("a", types.uint8, 3),  # 3 bits
        (None, types.uint8, 1),  # 1 bit padding
        ("b", types.uint8, 4),  # 4 bits
    ],
)


# ============================================================================
# EXACT COPIES FROM minimal_utcmma_1sm.cu
# ============================================================================

# -----------------------------------------------------------------------------
# TmemAddr (lines 97-103 in minimal_utcmma_1sm.cu)
# union TmemAddr{
#   uint32_t raw_addr_;
#   struct{
#     uint16_t col_id;   // bit [0, 16): col_id
#     uint16_t row_id;   // bit [16,32): row_id
#   };
# };
# -----------------------------------------------------------------------------

TmemAddrFields = struct(
    [
        ("col_id", types.uint16),  # bit [0, 16): col_id
        ("row_id", types.uint16),  # bit [16,32): row_id
    ],
)

TmemAddr = union(
    [
        ("raw_addr_", types.uint32),
        ("fields", TmemAddrFields),
    ],
)


# -----------------------------------------------------------------------------
# InstrDescriptor (lines 105-131 in minimal_utcmma_1sm.cu)
# union InstrDescriptor
# {
#   uint32_t desc_;
#
#   struct {
#     // Bitfield implementation avoids the need for shifts in assignment
#     uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
#              sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for HMMA/IMMA/QMMA
#              saturate_      : 1,  // bit [ 3, 4) : 0 = no saturate. 1 = saturate. 1 value valid only for IMMA
#              c_format_      : 2,  // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
#              sparse_format_ : 1,  // bit [ 6, 7) : 0 = TID, 1 = REGOFFSET (used only when sparse bit is set to 1)
#              a_format_      : 3,  // bit [ 7,10) : QMMA:0 = E4M3, 1 = E5M2, 2 = E3M4, 3 = E2M3, 4 = E3M2, 5 = E2M1. HMMA: 0 = F16, 1 = BF16, 2 = TF32, 3 = E6M9. IMMA: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
#              b_format_      : 3,  // bit [10,13) : QMMA:0 = E4M3, 1 = E5M2, 2 = E3M4, 3 = E2M3, 4 = E3M2, 5 = E2M1. HMMA: 0 = F16, 1 = BF16, 2 = TF32, 3 = E6M9. IMMA: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
#              a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for HMMA and QMMA
#              b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for HMMA and QMMA
#              a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16, E6M9 and TF32 source formats
#     uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16, E6M9 and TF32 source formats
#              n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
#                             : 1,  //
#              m_dim_         : 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
#                             : 1,  //
#              max_shift_     : 2;  // bit [30,32) : Maximum shift for WS instruction. Encoded as follows: 0 = no shift, 1 = maximum shift of 8, 2 = maximum shift of 16, 3 = maximum shift of 32.
#   };
# };
# -----------------------------------------------------------------------------

InstrDescriptorSimple = struct(
    [
        ("sparse_id2_", types.uint32, 2),  # bits [0,2)
        ("sparse_flag_", types.uint32, 1),  # bit  [2,3)
        ("saturate_", types.uint32, 1),  # bit  [3,4)
        ("c_format_", types.uint32, 2),  # bits [4,6)
        ("a_format_", types.uint32, 3),  # bits [7,10)
        ("b_format_", types.uint32, 3),  # bits [10,13)
    ],
    name="InstrDescriptorSimple",
)

InstrDescriptorSimpleUnion = union(
    [
        ("desc_", types.uint32),
        ("bitfields", InstrDescriptorSimple),
    ],
    name="InstrDescriptorSimpleUnion",
)

InstrDescriptorBitfields = struct(
    [
        # First uint16_t (bits 0-15)
        ("sparse_id2_", types.uint16, 2),  # bit [ 0, 2)
        ("sparse_flag_", types.uint16, 1),  # bit [ 2, 3)
        ("saturate_", types.uint16, 1),  # bit [ 3, 4)
        ("c_format_", types.uint16, 2),  # bit [ 4, 6)
        ("sparse_format_", types.uint16, 1),  # bit [ 6, 7)
        ("a_format_", types.uint16, 3),  # bit [ 7,10)
        ("b_format_", types.uint16, 3),  # bit [10,13)
        ("a_negate_", types.uint16, 1),  # bit [13,14)
        ("b_negate_", types.uint16, 1),  # bit [14,15)
        ("a_major_", types.uint16, 1),  # bit [15,16)
        # Second uint16_t (bits 16-31)
        ("b_major_", types.uint16, 1),  # bit [16,17)
        ("n_dim_", types.uint16, 6),  # bit [17,23)
        (None, types.uint16, 1),  # bit [23,24) - padding
        ("m_dim_", types.uint16, 5),  # bit [24,29)
        (None, types.uint16, 1),  # bit [29,30) - padding
        ("max_shift_", types.uint16, 2),  # bit [30,32)
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


# -----------------------------------------------------------------------------
# SmemDescriptor (lines 143-173 in minimal_utcmma_1sm.cu)
# union SmemDescriptor {
#   uint64_t desc_;
#   uint32_t reg32_[2];
#   uint16_t reg16_[4];
#
#   // Bitfield implementation avoids the need for shifts in assignment
#   struct {
#     // start_address, bit [0,14), 4LSB not included
#     uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
#     // leading dimension byte offset, bit [16,30), 4LSB not included
#     // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
#     //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
#     // For T: This is the stride from the first 8 rows to the next 8 rows.
#     uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
#     // stride dimension byte offset, bit [32,46), 4LSB not included
#     // For N: This is the stride from the first 8 rows to the next 8 rows.
#     // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
#     uint16_t stride_byte_offset_ : 14, version_ : 2; // 14 bits [0,14), 2 bits unused
#     // base_offset, bit [49,52)
#     // Valid only for SWIZZLE_128B and SWIZZLE_64B
#     uint8_t : 1, base_offset_ : 3, : 4; // 1 bit unused, 3 bits [1,4), 4 bits unused
#     // layout type, bit [61,64),
#     // SWIZZLE_NONE matrix descriptor = 0,
#     // SWIZZLE_128B matrix descriptor = 2,
#     // SWIZZLE_64B descriptor = 4,
#     // SWIZZLE_32B descriptor = 6,
#     // SWIZZLE_128B_BASE32B = 1,
#     // N/A = 3, N/A = 5, N/A = 7
#     uint8_t : 5, layout_type_ : 3; // 6 bits unused, 3 bits [5,8)
#   };
# };
# -----------------------------------------------------------------------------

SmemDescriptorSimple = struct(
    [
        ("start_address_", types.uint64, 14),  # bits [0,14)
        ("leading_byte_offset_", types.uint64, 14),  # bits [16,30)
        ("stride_byte_offset_", types.uint64, 14),  # bits [32,46)
        ("base_offset_", types.uint64, 3),  # bits [49,52)
        ("layout_type_", types.uint64, 3),  # bits [61,64)
    ],
    "SmemDescriptorSimple",
)

SmemDescriptorSimpleUnion = union(
    [
        ("desc_", types.uint64),
        ("bitfields", SmemDescriptorSimple),
    ],
    "SmemDescriptorSimpleUnion",
)

SmemDescriptorBitfields = struct(
    [
        # First uint16_t (bits 0-15)
        (
            "start_address_",
            types.uint16,
            14,
        ),  # start_address, bit [0,14), 4LSB not included
        (None, types.uint16, 2),  # 2 bits unused
        # Second uint16_t (bits 16-31)
        (
            "leading_byte_offset_",
            types.uint16,
            14,
        ),  # leading dimension byte offset, bit [16,30), 4LSB not included
        (None, types.uint16, 2),  # 2 bits unused
        # Third uint16_t (bits 32-47)
        (
            "stride_byte_offset_",
            types.uint16,
            14,
        ),  # stride dimension byte offset, bit [32,46), 4LSB not included
        ("version_", types.uint16, 2),  # bits [46,48)
        # Fourth uint8_t (bits 48-55)
        (None, types.uint8, 1),  # 1 bit unused
        (
            "base_offset_",
            types.uint8,
            3,
        ),  # base_offset, bit [49,52), Valid only for SWIZZLE_128B and SWIZZLE_64B
        (None, types.uint8, 4),  # 4 bits unused
        # Fifth uint8_t (bits 56-63)
        (None, types.uint8, 5),  # 5 bits unused
        (
            "layout_type_",
            types.uint8,
            3,
        ),  # layout type, bit [61,64): SWIZZLE_NONE=0, SWIZZLE_128B=2, SWIZZLE_64B=4, SWIZZLE_32B=6
    ],
    "SmemDescriptorBitfields",
)

SmemDescriptor = union(
    [
        ("desc_", types.uint64),
        ("reg32_", types.UniTuple(types.uint32, 2)),
        ("reg16_", types.UniTuple(types.uint16, 4)),
        ("bitfields", SmemDescriptorBitfields),
    ],
    "SmemDescriptor",
)


# ============================================================================
# Simple Type-Punning Unions
# ============================================================================

Uint32FloatUnion = union(
    [
        ("as_int", types.uint32),
        ("as_float", types.float32),
    ],
    name="Uint32FloatUnion",
)


# ============================================================================
# Array View Unions
# ============================================================================

Uint64Uint32ArrayUnion = union(
    [
        ("as_uint64", types.uint64),
        ("as_uint32_array", types.UniTuple(types.uint32, 2)),
    ],
)

Uint64Uint16ArrayUnion = union(
    [
        ("as_uint64", types.uint64),
        ("as_uint16_array", types.UniTuple(types.uint16, 4)),
    ],
)

MultiViewUnion = union(
    [
        ("as_uint64", types.uint64),
        ("as_uint32_array", types.UniTuple(types.uint32, 2)),
        ("as_uint16_array", types.UniTuple(types.uint16, 4)),
    ],
)


# ============================================================================
# Unions with Padded Bitfield Structs
# ============================================================================

PaddedBitfieldFields = struct(
    [
        ("x", types.uint32, 8),  # bits 0-7
        (None, types.uint32, 8),  # bits 8-15 (padding)
        ("y", types.uint32, 16),  # bits 16-31
    ],
)

PaddedBitfieldUnion = union(
    [
        ("raw", types.uint32),
        ("fields", PaddedBitfieldFields),
    ],
)


# ============================================================================
# SmemDescriptor with Bitfields for Array Views Test
# ============================================================================

SmemBitfieldsForArrayTest = struct(
    [
        # Lower 16 bits
        ("start_address", types.uint16, 14),
        (None, types.uint16, 2),  # padding
        # Next 16 bits
        ("leading_byte_offset", types.uint16, 14),
        (None, types.uint16, 2),  # padding
        # Next 16 bits
        ("stride_byte_offset", types.uint16, 14),
        ("version", types.uint16, 2),
        # Upper 16 bits
        (None, types.uint16, 1),  # padding
        ("base_offset", types.uint16, 3),
        (None, types.uint16, 4),  # padding
        (None, types.uint16, 5),  # padding
        ("layout_type", types.uint16, 3),
    ],
    "SmemBitfieldsForArrayTest",
)

SmemDescriptorForArrayTest = union(
    [
        ("desc_", types.uint64),
        ("reg32_", types.UniTuple(types.uint32, 2)),
        ("reg16_", types.UniTuple(types.uint16, 4)),
        ("bitfields", SmemBitfieldsForArrayTest),
    ],
    "SmemDescriptorForArrayTest",
)


# ============================================================================
# Bitfield Masking Test Struct
# ============================================================================

BitfieldMaskStruct = struct(
    [
        # First uint16_t (bits 0-15): 3+6+7 = 16 bits
        ("small_field", types.uint16, 3),  # bits [0,3) - 3-bit field (max value 7)
        ("medium_field", types.uint16, 6),  # bits [3,9) - 6-bit field (max value 63)
        (None, types.uint16, 7),  # bits [9,16) - padding
        # Second uint16_t (bits 16-31): 10+6 = 16 bits
        (
            "large_field",
            types.uint16,
            10,
        ),  # bits [16,26) - 10-bit field (max value 1023)
        (None, types.uint16, 6),  # bits [26,32) - padding
    ],
    "BitfieldMaskStruct",
)
