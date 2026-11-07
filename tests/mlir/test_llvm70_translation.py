# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from gpu_utils import requires_llvm70
from numba_cuda_mlir import tools
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir_optimization import _call_llvm70_capi


@requires_llvm70
def test_llvm70_translation_preserves_volatile_loads(monkeypatch):
    # Supply the target without requiring a CUDA device for an IR-only test.
    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda: "sm_90")
    with ir.Context():
        module = ir.Module.parse(
            """
            module {
              gpu.module @kernels {
                llvm.func @load_flags(%input: !llvm.ptr<1>, %output: !llvm.ptr<1>)
                    attributes {gpu.kernel} {
                  %used = llvm.load volatile %input : !llvm.ptr<1> -> i32
                  %unused0 = llvm.load volatile %input : !llvm.ptr<1> -> i32
                  %unused1 = llvm.load volatile %input : !llvm.ptr<1> -> i32
                  %ordinary = llvm.load %input : !llvm.ptr<1> -> i32
                  llvm.store volatile %used, %output : i32, !llvm.ptr<1>
                  llvm.store %ordinary, %output : i32, !llvm.ptr<1>
                  llvm.return
                }
              }
            }
            """
        )
        assert module.operation.verify()
        llvm_ir = _call_llvm70_capi(module, {"chip": "sm_90"}, gen_llvmir=True).decode()

    assert llvm_ir.count("load volatile i32, i32 addrspace(1)*") == 3
    assert llvm_ir.count("load i32, i32 addrspace(1)*") == 1
    assert llvm_ir.count("store volatile i32") == 1
    assert llvm_ir.count("store i32") == 1
    loads = [line for line in llvm_ir.splitlines() if " = load " in line]
    assert ["load volatile" in line for line in loads] == [True, True, True, False]
