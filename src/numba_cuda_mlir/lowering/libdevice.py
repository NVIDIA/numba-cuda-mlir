# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.lowering_utilities import int_of
from numba_cuda_mlir.lowering_utilities import convert_tuple_like
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.errors import InternalCompilerError
from numba_cuda_mlir.mlir_lowering import MLIRLower
from textwrap import dedent
from numba_cuda_mlir.lowering_utilities import convert, tensor_to_memref
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda import typing as typing
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import func, llvm, tensor, memref
from numba_cuda_mlir.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
lower = registry.lower
from numba_cuda_mlir.numba_cuda.extending import overload, intrinsic
import numba_cuda_mlir.cuda.libdevice as libdevice
import numba_cuda_mlir.cuda.libdevicefuncs as libdevicefuncs
import inspect
import ast
from numba_cuda_mlir.lowering_utilities import get_or_insert_function
from numba_cuda_mlir.numba_cuda.typing.templates import ConcreteTemplate
from typing import Callable, Any

_libdevice_descriptors: dict[Callable[..., Any], libdevicefuncs.Descriptor] = {}


def _generalize_type(ty: types.Type) -> types.Type:
    """
    Generalize types.int64 -> types.Integer(), recursively.
    """
    match ty:
        case types.UniTuple() as ut:
            return types.UniTuple(_generalize_type(ut.dtype), ut.count)
        case types.Tuple() as t:
            return types.Tuple(_generalize_type(ty) for ty in t.types)
        case types.Integer():
            return types.Integer
        case types.Float():
            return types.Float
        case _:
            raise NotImplementedError(f"Unknown type: {ty}")


def libdevice_implement_by_pointer(
    pyfunc, py_api: typing.Signature, c_api: typing.Signature
):
    def core(lower: MLIRLower, target, args, kwargs):
        trace("lower: %s", pyfunc.__name__)
        args = [lower.load_var(arg) for arg in args]
        by_ptr_return_types = [
            ty for ty in c_api.args if isinstance(ty, types.CPointer)
        ]
        by_ptr_return_mlir_types = [
            lower.get_mlir_type(ty.dtype) for ty in by_ptr_return_types
        ]
        by_ptr_return_vars = [
            lower.alloca(mlir_type) for mlir_type in by_ptr_return_mlir_types
        ]
        args += by_ptr_return_vars
        descriptor = _libdevice_descriptors[pyfunc]
        fn_type: ir.FunctionType = lower.get_mlir_type(descriptor.c_sig)
        libdevice_name = "__nv_" + pyfunc.__name__
        callee = get_or_insert_function(libdevice_name, fn_type, lower.mlir_gpu_module)
        result = [] if c_api.return_type is types.void else [fn_type.results[0]]
        call = func.call(result=result, callee=callee.name.value, operands_=args)
        result = [] if c_api.return_type is types.void else [call]
        result += [
            llvm.load(ty, var)
            for ty, var in zip(by_ptr_return_mlir_types, by_ptr_return_vars)
        ]
        lower.store_var(target, tuple(result))

    core.__name__ = f"lower_{pyfunc.__name__}_by_pointer"
    py_args = tuple(_generalize_type(ty) for ty in py_api.args)
    lower(pyfunc, *py_args)(core)


def libdevice_implement_uniform(pyfunc, api):
    def core(lower, target, args, kwargs):
        args = [lower.load_var(arg) for arg in args]
        trace("%s", pyfunc.__name__)
        descriptor = _libdevice_descriptors[pyfunc]
        fn_type: ir.FunctionType = lower.get_mlir_type(descriptor.c_sig)
        libdevice_name = "__nv_" + pyfunc.__name__
        callee = get_or_insert_function(libdevice_name, fn_type, lower.mlir_gpu_module)
        args = [convert(arg, ty) for arg, ty in zip(args, fn_type.inputs)]
        call = func.call(
            result=[fn_type.results[0]], callee=callee.name.value, operands_=args
        )
        call = convert(call, lower.get_mlir_type(target))
        lower.store_var(target, call)

    core.__name__ = f"lower_{pyfunc.__name__}_uniform"
    arg_types = tuple(_generalize_type(ty) for ty in api.args)
    lower(pyfunc, *arg_types)(core)


def libdevice_implement(pyfunc, py_api, c_api):
    if py_api == c_api:
        libdevice_implement_uniform(pyfunc, py_api)
    else:
        libdevice_implement_by_pointer(pyfunc, py_api, c_api)


def _libdevice_register():
    """
    Register type declarations and lowerings for all libdevice functions.
    """
    for descriptor in libdevicefuncs.libdevice_descriptors():
        py_sig = descriptor.py_sig
        c_sig = descriptor.c_sig
        pyfunc = getattr(libdevice, descriptor.py_name)
        _libdevice_descriptors[pyfunc] = descriptor
        libdevice_implement(pyfunc, py_sig, c_sig)


_libdevice_register()
