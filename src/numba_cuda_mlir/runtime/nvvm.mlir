// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Plain bindings to the NVVM dialect
// NOTE: this is just a convenience for ninja users--we need to
// add proper bindings under the numba_cuda_mlir.cuda module, but this allows
// for quick prototyping.
module attributes {numba_cuda_mlir.link_target} {
    func.func private @elect_sync() -> i1 attributes {always_inline} {
        %res = nvvm.elect.sync -> i1
        return %res : i1
    }
    func.func private @elect_sync_mask(%mask: i32) -> i1 attributes {always_inline} {
        %res = nvvm.elect.sync %mask -> i1
        return %res : i1
    }
    func.func private @mbarrier_init(%mbar : !llvm.ptr, %count: i32) attributes {always_inline} {
        nvvm.mbarrier.init %mbar, %count : !llvm.ptr, i32
        return
    }
    func.func private @mbarrier_try_wait(%mbar : !llvm.ptr, %phase: i32) attributes {always_inline} {
        // TODO(ajm): map the parity in a more convenient way
        %ticks = arith.constant 10000000 : i32
        nvvm.mbarrier.try_wait.parity %mbar, %phase, %ticks : !llvm.ptr, i32, i32
        return
    }
    func.func private @barrier() attributes {always_inline} {
        nvvm.barrier
        return
    }
    func.func private @mbarrier_arrive_expect_tx(%mbar : !llvm.ptr, %txcount: i32) attributes {always_inline} {
        nvvm.mbarrier.arrive.expect_tx %mbar, %txcount : !llvm.ptr, i32
        return
    }
    func.func private @cp_async_bulk_load(%dst_: !llvm.ptr, %src_: !llvm.ptr, %mbar_: !llvm.ptr, %bytes : i32) attributes {always_inline} {
        %dst = llvm.addrspacecast %dst_ : !llvm.ptr to !llvm.ptr<7>
        %src = llvm.addrspacecast %src_ : !llvm.ptr to !llvm.ptr<1>
        %mbar = llvm.addrspacecast %mbar_ : !llvm.ptr to !llvm.ptr<3>
        nvvm.cp.async.bulk.shared.cluster.global %dst, %src, %mbar, %bytes : !llvm.ptr<7>, !llvm.ptr<1>
        return
    }
    func.func private @fence_mbarrier_init() attributes {always_inline} {
        nvvm.fence.mbarrier.init
        return
    }
    func.func private @cp_async_bulk_store(%dst_: !llvm.ptr, %src_: !llvm.ptr, %bytes : i32) attributes {always_inline} {
        %dst = llvm.addrspacecast %dst_ : !llvm.ptr to !llvm.ptr<1>
        %src = llvm.addrspacecast %src_ : !llvm.ptr to !llvm.ptr<3>
        nvvm.cp.async.bulk.global.shared.cta %dst, %src, %bytes : !llvm.ptr<1>, !llvm.ptr<3>
        return
    }

    // TMA bulk tensor load (2D) - single CTA
    func.func private @cp_async_bulk_tensor_2d(
        %dst_: !llvm.ptr,
        %tma_desc_: !llvm.ptr,
        %coord0: i32,
        %coord1: i32,
        %mbar_: !llvm.ptr
    ) attributes {always_inline} {
        %dst = llvm.addrspacecast %dst_ : !llvm.ptr to !llvm.ptr<7>
        %mbar = llvm.addrspacecast %mbar_ : !llvm.ptr to !llvm.ptr<3>
        nvvm.cp.async.bulk.tensor.shared.cluster.global %dst, %tma_desc_, %mbar, box[%coord0, %coord1]
            {mode = #nvvm.tma_load_mode<tile>, isCTAOnly = false, group = #nvvm.cta_group<cta_1>}
            : !llvm.ptr<7>, !llvm.ptr
        return
    }

    // TMA bulk tensor load (2D) with multicast - 2 CTA cluster
    func.func private @cp_async_bulk_tensor_2d_multicast(
        %dst_: !llvm.ptr,
        %tma_desc_: !llvm.ptr,
        %coord0: i32,
        %coord1: i32,
        %mbar_: !llvm.ptr,
        %multicast_mask: i16
    ) attributes {always_inline} {
        %dst = llvm.addrspacecast %dst_ : !llvm.ptr to !llvm.ptr<7>
        %mbar = llvm.addrspacecast %mbar_ : !llvm.ptr to !llvm.ptr<3>
        nvvm.cp.async.bulk.tensor.shared.cluster.global %dst, %tma_desc_, %mbar, box[%coord0, %coord1]
            multicast_mask = %multicast_mask
            {mode = #nvvm.tma_load_mode<tile>, isCTAOnly = false, group = #nvvm.cta_group<cta_2>}
            : !llvm.ptr<7>, !llvm.ptr
        return
    }

    // Cluster operations
    func.func private @cluster_arrive_relaxed() attributes {always_inline} {
        nvvm.cluster.arrive.relaxed
        return
    }
    func.func private @cluster_wait() attributes {always_inline} {
        nvvm.cluster.wait
        return
    }
    func.func private @block_idx_in_cluster() -> i32 attributes {always_inline} {
        %res = nvvm.read.ptx.sreg.clusterid.x : i32
        return %res : i32
    }

    // CTA rank within the cluster (0 to cluster_size-1)
    func.func private @cluster_ctarank() -> i32 attributes {always_inline} {
        %res = nvvm.read.ptx.sreg.cluster.ctarank : i32
        return %res : i32
    }

    // Map shared memory pointer to remote CTA within cluster
    // Returns a pointer in shared cluster address space (7) that can be used
    // to access the same shared memory location from any CTA in the cluster
    func.func private @mapa(%ptr: !llvm.ptr, %remote_cta: i32) -> !llvm.ptr attributes {always_inline} {
        %ptr_shared = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<3>
        %result = nvvm.mapa %ptr_shared, %remote_cta : !llvm.ptr<3> -> !llvm.ptr<7>
        %result_generic = llvm.addrspacecast %result : !llvm.ptr<7> to !llvm.ptr
        return %result_generic : !llvm.ptr
    }

    // Warp operations
    func.func private @warp_idx() -> i32 attributes {always_inline} {
        %tid = nvvm.read.ptx.sreg.tid.x : i32
        %warp_size = arith.constant 32 : i32
        %warp_id = arith.divui %tid, %warp_size : i32
        return %warp_id : i32
    }


    // =========================================================================
    // Prefetch operations
    // =========================================================================

    // Prefetch TMA descriptor into cache
    func.func private @prefetch_tensormap(%tma_desc: !llvm.ptr) attributes {always_inline} {
        nvvm.prefetch %tma_desc {tensormap} : !llvm.ptr
        return
    }
}
