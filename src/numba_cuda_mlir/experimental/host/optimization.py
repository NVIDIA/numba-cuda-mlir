# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.passmanager import PassManager
from numba_cuda_mlir.lowering_utilities import context as mlir_context
from numba_cuda_mlir.mlir_lowering import get_mlir_module_str


TO_LLVM_DIALECT_PIPELINE = """
    builtin.module(
      reconcile-unrealized-casts,
      convert-scf-to-cf,
      convert-math-to-llvm,
      finalize-memref-to-llvm{index-bitwidth=64},
      convert-arith-to-llvm{index-bitwidth=0},
      convert-index-to-llvm{index-bitwidth=64},
      convert-cf-to-llvm{index-bitwidth=0},
      convert-func-to-llvm{index-bitwidth=0},
      canonicalize,
      cse,
      reconcile-unrealized-casts
    )
    """


def lower_to_llvm_dialect(cres):
    with mlir_context.get_context():
        module = ir.Module.parse(get_mlir_module_str(cres.metadata))
        options = cres.metadata["targetoptions"]
        if options.get("dump_mlir") or options.get("dump"):
            print(f"=============== Host MLIR ===============\n\n{module}\n")
        PassManager.parse(TO_LLVM_DIALECT_PIPELINE).run(module.operation)
        if options.get("dump_llvm_dialect") or options.get("dump"):
            print(f"=============== Host LLVM-Dialect MLIR ===============\n\n{module}\n")
        return module
