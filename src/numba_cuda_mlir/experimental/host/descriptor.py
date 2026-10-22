# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import cached_property, update_wrapper

from numba_cuda_mlir.descriptor import MLIRTargetContext, MLIRTypingContext
from numba_cuda_mlir.experimental.host.execution import scalar_ctype
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import sigutils
from numba_cuda_mlir.numba_cuda.core.compiler_lock import global_compiler_lock
from numba_cuda_mlir.numba_cuda.core.target_extension import CPU, target_registry
from numba_cuda_mlir.numba_cuda.typing.typeof import Purpose, typeof


class TargetDescriptor:
    @cached_property
    def typing_context(self):
        return MLIRTypingContext()

    @cached_property
    def target_context(self):
        return MLIRTargetContext(self.typing_context, target="numba_cuda_mlir_host")


class Target(CPU):
    pass


target_registry["numba_cuda_mlir_host"] = Target
target = TargetDescriptor()


class Dispatcher:
    def __init__(self, py_func, targetoptions):
        self.py_func = py_func
        self.targetoptions = targetoptions
        self.overloads = {}
        self._can_compile = True
        self._signature = inspect.signature(py_func)
        self._arg_annotations = tuple(
            parameter.annotation for parameter in self._signature.parameters.values()
        )
        for parameter in self._signature.parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                raise TypeError("host.jit supports only fixed positional parameters")
        for annotation in self._arg_annotations:
            if annotation is not inspect.Signature.empty:
                if not isinstance(annotation, types.Type):
                    raise TypeError("host.jit annotations must be Numba scalar types")
                scalar_ctype(annotation)
        update_wrapper(self, py_func)

    @global_compiler_lock
    def compile(self, sig):
        from numba_cuda_mlir.experimental.host.compiler import compile

        argtypes, return_type = sigutils.normalize_signature(sig)
        argtypes = tuple(argtypes)
        if len(argtypes) != len(self._arg_annotations):
            raise TypeError(
                f"expected {len(self._arg_annotations)} argument types, got {len(argtypes)}"
            )
        if argtypes in self.overloads:
            return self.overloads[argtypes]
        if not self._can_compile:
            raise TypeError(f"no matching host.jit signature for {argtypes}")
        result = compile(self.py_func, return_type, argtypes, self.targetoptions)
        self.overloads[argtypes] = result
        return result

    def disable_compile(self):
        self._can_compile = False

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError("host.jit does not support keyword arguments")
        if len(args) != len(self._arg_annotations):
            raise TypeError(f"expected {len(self._arg_annotations)} arguments, got {len(args)}")
        # Validate actual scalars before annotations select the specialization;
        # execution owns their subsequent Python-to-native slot conversion.
        actual_types = tuple(typeof(value, Purpose.argument) for value in args)
        for typ in actual_types:
            scalar_ctype(typ)
        argtypes = tuple(
            annotation if isinstance(annotation, types.Type) else actual_type
            for annotation, actual_type in zip(self._arg_annotations, actual_types, strict=True)
        )
        result = self.compile(argtypes)
        return result.entry_point(*args)

    @property
    def signatures(self):
        return list(self.overloads)

    def inspect_mlir(self, signature=None):
        if signature is None:
            return {sig: result.mlir for sig, result in self.overloads.items()}
        argtypes, _ = sigutils.normalize_signature(signature)
        return self.overloads[tuple(argtypes)].mlir

    def inspect_llvm_dialect(self, signature=None):
        if signature is None:
            return {sig: result.llvm_mlir for sig, result in self.overloads.items()}
        argtypes, _ = sigutils.normalize_signature(signature)
        return self.overloads[tuple(argtypes)].llvm_mlir
