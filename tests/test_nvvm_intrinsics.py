# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import compiler, testing, typing, tools
import ctypes
import pytest

cc = tools.get_gpu_compute_capability(tuple)


# @pytest.mark.skip()
@pytest.mark.skipif(cc < (10, 0), reason="sm_100 or greater is required for this test")
def test_nvvm_intrinsics():
    pytest.importorskip("torch")
    import torch

    size = 16384
    a = torch.ones(size).to(torch.float32).cuda()  # 4 bytes per float4
    b = torch.zeros(size).to(torch.float32).cuda()  # 4 bytes per float4
    grid = (size // 1024, 1, 1)
    block = (32, 1, 1)
    elems_per_blk = 1024
    bytes_per_blk = (4 * elems_per_blk) + 16

    @cuda.jit(opt_level=3, features="+ptx80")
    def block_copy(a: torch.Tensor, b: torch.Tensor):
        tx = cuda.threadIdx.x
        ap = ctypes.pointer(a)
        bp = ctypes.pointer(b)
        smem = cuda.shared_array(shape=(elems_per_blk,), dtype=a.dtype)
        mbar = cuda.shared_array(shape=(1,), dtype=torch.int64)

        smemp = ctypes.pointer(smem)
        mbarp = ctypes.pointer(mbar)

        if cuda.intrin.elect_sync():
            cuda.intrin.mbarrier_init(mbarp, 1)

        cuda.intrin.fence_mbarrier_init()
        cuda.intrin.barrier()

        if cuda.intrin.elect_sync():
            cuda.intrin.mbarrier_arrive_expect_tx(mbarp, bytes_per_blk)
            cuda.intrin.cp_async_bulk_load(smemp, ap, mbarp, bytes_per_blk)

        cuda.intrin.mbarrier_try_wait(mbarp, 0)

        if cuda.intrin.elect_sync():
            cuda.intrin.cp_async_bulk_store(bp, smemp, bytes_per_blk)

    block_copy[grid, block, 0, bytes_per_blk](a, b)

    cres = compiler.compile_for(block_copy, a, b)

    testing.filecheck_with_comments(cres.mlir_module_optimized)
    # CHECK-LABEL:     llvm.func @_ZN20test_nvvm_intrinsics12_3clocals_3e10block_copyE5ArrayIfLi1E1C7mutable7alignedE5ArrayIfLi1E1C7mutable7alignedE(
    # CHECK-SAME:      %{{.*}}: i64) attributes {numba_cuda_mlir.arg_attrs = [{}, {}], numba_cuda_mlir.orig_arg_types = [memref<?xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>], gpu.kernel, llvm.emit_c_interface, nvvm.kernel} {
    # CHECK:             %{{.*}} = llvm.mlir.constant(10000000 : i32) : i32
    # CHECK:             %{{.*}} = llvm.mlir.addressof @static_shared_memory_{{.*}} : !llvm.ptr<3>
    # CHECK:             %{{.*}} = llvm.mlir.addressof @static_shared_memory_{{.*}} : !llvm.ptr<3>
    # CHECK:             %{{.*}} = llvm.mlir.constant(1 : i32) : i32
    # CHECK:             %{{.*}} = llvm.mlir.constant(4112 : i32) : i32
    # CHECK:             %{{.*}} = llvm.mlir.constant(0 : i32) : i32
    # CHECK:             %{{.*}} = nvvm.elect.sync -> i1
    # CHECK:             llvm.cond_br %{{.*}}, ^bb1, ^bb2
    # CHECK:           ^bb1:
    # CHECK:             nvvm.mbarrier.init %{{.*}}, %{{.*}} : !llvm.ptr, i32
    # CHECK:             llvm.br ^bb2
    # CHECK:           ^bb2:
    # CHECK:             nvvm.fence.mbarrier.init
    # CHECK:             nvvm.barrier
    # CHECK:             %{{.*}} = nvvm.elect.sync -> i1
    # CHECK:             llvm.cond_br %{{.*}}, ^bb3, ^bb4
    # CHECK:           ^bb3:
    # CHECK:             nvvm.mbarrier.arrive.expect_tx %{{.*}}, %{{.*}} : !llvm.ptr, i32
    # CHECK:             nvvm.cp.async.bulk.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.ptr<7>, <1>
    # CHECK:             llvm.br ^bb4
    # CHECK:           ^bb4:
    # CHECK:             llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.b64 P1, [$0], $1, $2; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}", "l,r,r" %{{.*}}, %{{.*}}, %{{.*}} : (!llvm.ptr, i32, i32) -> ()
    # CHECK:             %{{.*}} = nvvm.elect.sync -> i1
    # CHECK:             llvm.cond_br %{{.*}}, ^bb5, ^bb6
    # CHECK:           ^bb5:
    # CHECK:             nvvm.cp.async.bulk.global.shared.cta %{{.*}}, %{{.*}}, %{{.*}} : <1>, <3>
    # CHECK:             llvm.return
    # CHECK:           ^bb6:
    # CHECK:             llvm.return
    # CHECK:           }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    test_nvvm_intrinsics()
