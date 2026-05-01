// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
module attributes {numba_cuda_mlir.link_target} {
    // tcgen05 intrinsics for UTCMMA operations
    // These wrap NVVM dialect operations for tcgen05 tensor core operations

    // =========================================================================
    // CTA_1 (single CTA) variants
    // =========================================================================
    
    // Allocate TMEM (Tensor Memory)
    func.func private @tcgen05_alloc(%ptr: !llvm.ptr, %columns: i32) attributes {always_inline} {
        %ptr3 = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<3>
        nvvm.tcgen05.alloc %ptr3, %columns {group = #nvvm.cta_group<cta_1>} : !llvm.ptr<3>, i32
        return
    }

    // tcgen05 MMA operation
    func.func private @tcgen05_mma(
        %tmem_addr: i32,
        %smem_desc_a: i64,
        %smem_desc_b: i64,
        %instr_desc: i32,
        %scale_d: i32
    ) attributes {always_inline} {
        %tmem_ptr = llvm.inttoptr %tmem_addr : i32 to !llvm.ptr<6>
        %c0_i32 = arith.constant 0 : i32
        %cmp = arith.cmpi ne, %scale_d, %c0_i32 : i32
        nvvm.tcgen05.mma %tmem_ptr, %smem_desc_a, %smem_desc_b, %instr_desc, %cmp
            {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_1>}
            : (!llvm.ptr<6>, i64, i64, i32, i1)
        return
    }

    // tcgen05 commit and mbarrier arrive
    func.func private @tcgen05_commit_arrive(%mbar_: !llvm.ptr) attributes {always_inline} {
        %mbar = llvm.addrspacecast %mbar_ : !llvm.ptr to !llvm.ptr<3>
        nvvm.tcgen05.commit %mbar {group = #nvvm.cta_group<cta_1>} : !llvm.ptr<3>
        return
    }

    // tcgen05 load from TMEM - loads a single f32 value
    func.func private @tcgen05_ld(%tmem_offset: i32) -> f32 attributes {always_inline} {
        %tmem_ptr = llvm.inttoptr %tmem_offset : i32 to !llvm.ptr<6>
        %result_i32 = nvvm.tcgen05.ld %tmem_ptr {shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2xi32>
        %c0 = arith.constant 0 : index
        %val_i32 = vector.extract %result_i32[%c0] : i32 from vector<2xi32>
        %val = arith.bitcast %val_i32 : i32 to f32
        return %val : f32
    }

    // tcgen05 load from TMEM - loads 2 f32 values (x2 load)
    // Returns both values from the vector<2xi32> result
    func.func private @tcgen05_ld_x2(%tmem_offset: i32) -> (f32, f32) attributes {always_inline} {
        %tmem_ptr = llvm.inttoptr %tmem_offset : i32 to !llvm.ptr<6>
        %result_i32 = nvvm.tcgen05.ld %tmem_ptr {shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>} : vector<2xi32>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %val0_i32 = vector.extract %result_i32[%c0] : i32 from vector<2xi32>
        %val1_i32 = vector.extract %result_i32[%c1] : i32 from vector<2xi32>
        %val0 = arith.bitcast %val0_i32 : i32 to f32
        %val1 = arith.bitcast %val1_i32 : i32 to f32
        return %val0, %val1 : f32, f32
    }

    // tcgen05 deallocate TMEM
    func.func private @tcgen05_dealloc(%tmem_addr: i32, %columns: i32) attributes {always_inline} {
        %tmem_ptr = llvm.inttoptr %tmem_addr : i32 to !llvm.ptr<6>
        nvvm.tcgen05.dealloc %tmem_ptr, %columns {group = #nvvm.cta_group<cta_1>} : !llvm.ptr<6>, i32
        return
    }

    // tcgen05 relinquish allocation permit
    func.func private @tcgen05_relinquish_alloc_permit() attributes {always_inline} {
        nvvm.tcgen05.relinquish_alloc_permit {group = #nvvm.cta_group<cta_1>}
        return
    }

    // =========================================================================
    // 2-CTA variants for cluster-wide TMEM operations
    // =========================================================================

    // Allocate TMEM (2-CTA cluster)
    func.func private @tcgen05_alloc_2cta(%ptr: !llvm.ptr, %columns: i32) attributes {always_inline} {
        %ptr3 = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<3>
        nvvm.tcgen05.alloc %ptr3, %columns {group = #nvvm.cta_group<cta_2>} : !llvm.ptr<3>, i32
        return
    }

    // tcgen05 MMA operation (2-CTA cluster)
    func.func private @tcgen05_mma_2cta(
        %tmem_addr: i32,
        %smem_desc_a: i64,
        %smem_desc_b: i64,
        %instr_desc: i32,
        %scale_d: i32
    ) attributes {always_inline} {
        %tmem_ptr = llvm.inttoptr %tmem_addr : i32 to !llvm.ptr<6>
        %c0_i32 = arith.constant 0 : i32
        %cmp = arith.cmpi ne, %scale_d, %c0_i32 : i32
        nvvm.tcgen05.mma %tmem_ptr, %smem_desc_a, %smem_desc_b, %instr_desc, %cmp
            {kind = #nvvm.tcgen05_mma_kind<f16>, ctaGroup = #nvvm.cta_group<cta_2>}
            : (!llvm.ptr<6>, i64, i64, i32, i1)
        return
    }

    // tcgen05 deallocate TMEM (2-CTA cluster)
    func.func private @tcgen05_dealloc_2cta(%tmem_addr: i32, %columns: i32) attributes {always_inline} {
        %tmem_ptr = llvm.inttoptr %tmem_addr : i32 to !llvm.ptr<6>
        nvvm.tcgen05.dealloc %tmem_ptr, %columns {group = #nvvm.cta_group<cta_2>} : !llvm.ptr<6>, i32
        return
    }

    // tcgen05 relinquish allocation permit (2-CTA cluster)
    func.func private @tcgen05_relinquish_alloc_permit_2cta() attributes {always_inline} {
        nvvm.tcgen05.relinquish_alloc_permit {group = #nvvm.cta_group<cta_2>}
        return
    }
}
