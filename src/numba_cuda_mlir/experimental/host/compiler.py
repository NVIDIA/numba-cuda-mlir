# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from numba_cuda_mlir.experimental.host.descriptor import target
from numba_cuda_mlir.experimental.host.execution import scalar_ctype
from numba_cuda_mlir.experimental.host.lowering import Lower
from numba_cuda_mlir.experimental.host.optimization import lower_to_llvm_dialect
from numba_cuda_mlir import mlir_compiler
from numba_cuda_mlir.mlir_lowering import get_mlir_module_str
from numba_cuda_mlir.numba_cuda import compiler, types, typing
from numba_cuda_mlir.numba_cuda.core.compiler_lock import global_compiler_lock
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.core.compiler_machinery import register_pass
from numba_cuda_mlir.numba_cuda.core.target_extension import target_override
from numba_cuda_mlir.numba_cuda.flags import CUDAFlags


@dataclass
class CompileResult:
    signature: typing.Signature
    fndesc: Any
    mlir: str
    llvm_mlir: str
    entry_point: Any
    execution_engine: Any
    metadata: dict[str, Any]


@register_pass(mutates_CFG=True, analysis_only=False)
class Backend(mlir_compiler.MLIRBackend):
    _name = "host_backend"

    def run_pass(self, state):
        if not isinstance(state.return_type, types.NoneType):
            try:
                scalar_ctype(state.return_type)
            except TypeError as exc:
                raise TypingError(str(exc)) from exc
        return super().run_pass(state)

    def create_lowerer(self, state, fndesc):
        return Lower(state.targetctx, fndesc, state.func_ir, state.metadata)


@global_compiler_lock
def compile_to_llvm_dialect(pyfunc, return_type, args, targetoptions):
    try:
        for arg in args:
            scalar_ctype(arg)
        if return_type is not None and not isinstance(return_type, types.NoneType):
            scalar_ctype(return_type)
    except TypeError as exc:
        raise TypingError(str(exc)) from exc
    typingctx = target.typing_context
    targetctx = target.target_context
    flags = CUDAFlags()
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    flags.nvvm_options = {}
    with target_override("numba_cuda_mlir_host"):
        typingctx.refresh()
        targetctx.refresh()
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=mlir_compiler.get_compiler_class(targetoptions, backend_pass=Backend),
        )
    return cres, lower_to_llvm_dialect(cres)


@global_compiler_lock
def compile(pyfunc, return_type, args, targetoptions):
    cres, module = compile_to_llvm_dialect(pyfunc, return_type, args, targetoptions)
    from numba_cuda_mlir.experimental.host.execution import ExecutionEngine

    engine = ExecutionEngine(
        module, cres.signature, cres.metadata["host_symbol_name"], targetoptions
    )
    return CompileResult(
        signature=cres.signature,
        fndesc=cres.fndesc,
        mlir=get_mlir_module_str(cres.metadata),
        llvm_mlir=str(module),
        entry_point=engine.get_callable(),
        execution_engine=engine,
        metadata=cres.metadata,
    )
