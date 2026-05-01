// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
module attributes {numba_cuda_mlir.link_target} {
    // Args match the bit-fields described in InstrDescriptor.
    // Each field is zero-extended, shifted into position, OR’d into the 32-bit result.
    //
    // union InstrDescriptor
    // {
    //   uint32_t desc_;
    //
    //   struct {
    //     // Bitfield implementation avoids the need for shifts in assignment
    //     uint16_t sparse_id2_    : 2,  // bit [ 0, 2) : Sparse meta data id2
    //              sparse_flag_   : 1,  // bit [ 2, 3) : 0 = dense. 1 = sparse. 1 value valid only for HMMA/IMMA/QMMA
    //              saturate_      : 1,  // bit [ 3, 4) : 0 = no saturate. 1 = saturate. 1 value valid only for IMMA
    //              c_format_      : 2,  // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
    //              sparse_format_ : 1,  // bit [ 6, 7) : 0 = TID, 1 = REGOFFSET (used only when sparse bit is set to 1)
    //              a_format_      : 3,  // bit [ 7,10) : QMMA:0 = E4M3, 1 = E5M2, 2 = E3M4, 3 = E2M3, 4 = E3M2, 5 = E2M1. HMMA: 0 = F16, 1 = BF16, 2 = TF32, 3 = E6M9. IMMA: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
    //              b_format_      : 3,  // bit [10,13) : QMMA:0 = E4M3, 1 = E5M2, 2 = E3M4, 3 = E2M3, 4 = E3M2, 5 = E2M1. HMMA: 0 = F16, 1 = BF16, 2 = TF32, 3 = E6M9. IMMA: 0 unsigned 8 bit, 1 signed 8 bit. BMMA: 0 Boolean
    //              a_negate_      : 1,  // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for HMMA and QMMA
    //              b_negate_      : 1,  // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for HMMA and QMMA
    //              a_major_       : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16, E6M9 and TF32 source formats
    //     uint16_t b_major_       : 1,  // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for E4M3, E5M2, INT8 (signed and unsigned), F16, BF16, E6M9 and TF32 source formats
    //              n_dim_         : 6,  // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32 (N=256).  All values are not valid for all instruction formats
    //                             : 1,  //
    //              m_dim_         : 5,  // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16 (M=256)
    //                             : 1,  //
    //              max_shift_     : 2;  // bit [30,32) : Maximum shift for WS instruction. Encoded as follows: 0 = no shift, 1 = maximum shift of 8, 2 = maximum shift of 16, 3 = maximum shift of 32.
    //   };
    //   // Decay to a uint32_t
    //   __host__ __device__ constexpr explicit
    //   operator uint32_t() const noexcept { return desc_; }
    // };
    //
    func.func private @encode_instr_descriptor(
        %sparse_id2        : i16,  // bits 0..1
        %sparse_flag       : i1,   // bit 2
        %saturate          : i1,   // bit 3
        %c_format          : i16,  // bits 4..5
        %sparse_format     : i1,   // bit 6
        %a_format          : i16,  // bits 7..9
        %b_format          : i16,  // bits 10..12
        %a_negate          : i1,   // bit 13
        %b_negate          : i1,   // bit 14
        %a_major           : i1,   // bit 15
        %b_major           : i1,   // bit 16
        %n_dim             : i16,  // bits 17..22
        %m_dim             : i16,  // bits 24..28
        %max_shift         : i16   // bits 30..31
    ) -> i32 attributes { always_inline } {
        %c2 = arith.constant 2 : i32
        %c3 = arith.constant 3 : i32
        %c4 = arith.constant 4 : i32
        %c6 = arith.constant 6 : i32
        %c7 = arith.constant 7 : i32
        %c10 = arith.constant 10 : i32
        %c13 = arith.constant 13 : i32
        %c14 = arith.constant 14 : i32
        %c15 = arith.constant 15 : i32
        %c16 = arith.constant 16 : i32
        %c17 = arith.constant 17 : i32
        %c24 = arith.constant 24 : i32
        %c30 = arith.constant 30 : i32
        %v0 = arith.extui %sparse_id2 : i16 to i32
        %v1 = arith.extui %sparse_flag : i1 to i32
        %v2 = arith.shli %v1, %c2 : i32
        %v3 = arith.ori %v0, %v2 : i32

        %v4 = arith.extui %saturate : i1 to i32
        %v5 = arith.shli %v4, %c3 : i32
        %v6 = arith.ori %v3, %v5 : i32

        %v7 = arith.extui %c_format : i16 to i32
        %v8 = arith.shli %v7, %c4 : i32
        %v9 = arith.ori %v6, %v8 : i32

        %v10 = arith.extui %sparse_format : i1 to i32
        %v11 = arith.shli %v10, %c6 : i32
        %v12 = arith.ori %v9, %v11 : i32

        %v13 = arith.extui %a_format : i16 to i32
        %v14 = arith.shli %v13, %c7 : i32
        %v15 = arith.ori %v12, %v14 : i32

        %v16 = arith.extui %b_format : i16 to i32
        %v17 = arith.shli %v16, %c10 : i32
        %v18 = arith.ori %v15, %v17 : i32

        %v19 = arith.extui %a_negate : i1 to i32
        %v20 = arith.shli %v19, %c13 : i32
        %v21 = arith.ori %v18, %v20 : i32

        %v22 = arith.extui %b_negate : i1 to i32
        %v23 = arith.shli %v22, %c14 : i32
        %v24 = arith.ori %v21, %v23 : i32

        %v25 = arith.extui %a_major : i1 to i32
        %v26 = arith.shli %v25, %c15 : i32
        %v27 = arith.ori %v24, %v26 : i32

        %v28 = arith.extui %b_major : i1 to i32
        %v29 = arith.shli %v28, %c16 : i32
        %v30 = arith.ori %v27, %v29 : i32

        %v31 = arith.extui %n_dim : i16 to i32
        %v32 = arith.shli %v31, %c17 : i32
        %v33 = arith.ori %v30, %v32 : i32

        %v34 = arith.extui %m_dim : i16 to i32
        %v35 = arith.shli %v34, %c24 : i32
        %v36 = arith.ori %v33, %v35 : i32

        %v37 = arith.extui %max_shift : i16 to i32
        %v38 = arith.shli %v37, %c30 : i32
        %v39 = arith.ori %v36, %v38 : i32
        return %v39 : i32
    }

    // Utility function to encode the shared memory descriptor
    //
    // union SmemDescriptor {
    //   uint64_t desc_;
    //   uint32_t reg32_[2];
    //   uint16_t reg16_[4];
    //
    //   // Bitfield implementation avoids the need for shifts in assignment
    //   struct {
    //     // start_address, bit [0,14), 4LSB not included
    //     uint16_t start_address_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    //     // leading dimension byte offset, bit [16,30), 4LSB not included
    //     // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
    //     //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    //     // For T: This is the stride from the first 8 rows to the next 8 rows.
    //     uint16_t leading_byte_offset_ : 14, : 2; // 14 bits [0,14), 2 bits unused
    //     // stride dimension byte offset, bit [32,46), 4LSB not included
    //     // For N: This is the stride from the first 8 rows to the next 8 rows.
    //     // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    //     uint16_t stride_byte_offset_ : 14, version_ : 2; // 14 bits [0,14), 2 bits unused
    //     // base_offset, bit [49,52)
    //     // Valid only for SWIZZLE_128B and SWIZZLE_64B
    //     uint8_t : 1, base_offset_ : 3, : 4; // 1 bit unused, 3 bits [1,4), 4 bits unused
    //     // layout type, bit [61,64),
    //     // SWIZZLE_NONE matrix descriptor = 0,
    //     // SWIZZLE_128B matrix descriptor = 2,
    //     // SWIZZLE_64B descriptor = 4,
    //     // SWIZZLE_32B descriptor = 6,
    //     // SWIZZLE_128B_BASE32B = 1,
    //     // N/A = 3, N/A = 5, N/A = 7
    //     uint8_t : 5, layout_type_ : 3; // 6 bits unused, 3 bits [5,8)
    //   };
    // };
    //
    func.func private @encode_smem_descriptor(
        %start_address        : i16,  // bits 0..13
        %leading_byte_offset  : i16,  // bits 16..29 (2 bits padding)
        %stride_byte_offset   : i16,  // bits 32..45 (2 bits padding)
        %base_offset          : i8,   // bits 49..51 (1 bit padding before, 4 after)
        %layout_type          : i8    // bits 61..63 (5 bits padding)
    ) -> i64 attributes { always_inline } {
        %c0_i64 = arith.constant 0 : i64
        %mask14 = arith.constant 16383 : i64  // 14-bit mask
        %mask3 = arith.constant 7 : i64      // 3-bit mask
        %c16 = arith.constant 16 : i64
        %c32 = arith.constant 32 : i64
        %c49 = arith.constant 49 : i64
        %c61 = arith.constant 61 : i64

        // start_address occupies bits [0, 13]
        %start_i64 = arith.extui %start_address : i16 to i64
        %start = arith.andi %start_i64, %mask14 : i64
        %result0 = arith.ori %c0_i64, %start : i64

        // leading_byte_offset occupies bits [16, 29]
        %leading_i64 = arith.extui %leading_byte_offset : i16 to i64
        %leading = arith.andi %leading_i64, %mask14 : i64
        %leading_shifted = arith.shli %leading, %c16 : i64
        %result1 = arith.ori %result0, %leading_shifted : i64

        // stride_byte_offset occupies bits [32, 45]
        %stride_i64 = arith.extui %stride_byte_offset : i16 to i64
        %stride = arith.andi %stride_i64, %mask14 : i64
        %stride_shifted = arith.shli %stride, %c32 : i64
        %result2 = arith.ori %result1, %stride_shifted : i64

        // base_offset occupies bits [49, 51]
        %base_i64 = arith.extui %base_offset : i8 to i64
        %base = arith.andi %base_i64, %mask3 : i64
        %base_shifted = arith.shli %base, %c49 : i64
        %result3 = arith.ori %result2, %base_shifted : i64

        // layout_type occupies bits [61, 63]
        %layout_i64 = arith.extui %layout_type : i8 to i64
        %layout = arith.andi %layout_i64, %mask3 : i64
        %layout_shifted = arith.shli %layout, %c61 : i64
        %result4 = arith.ori %result3, %layout_shifted : i64

        return %result4 : i64
    }
}
