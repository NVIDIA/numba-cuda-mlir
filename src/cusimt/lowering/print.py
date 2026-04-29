# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator
from cusimt.lowering_utilities import index_of
from cusimt.numba_cuda import types
from cusimt.numba_cuda.types.ext_types import Dim3
from cusimt.mlir_lowering import MLIRLower
from cusimt._mlir import ir
from cusimt._mlir.extras import types as T
from cusimt._mlir.dialects import gpu, memref, arith
from cusimt.mlir.dialect_exts import scf
from cusimt.mlir.dialect_exts.scf import (
    if_ctx_manager as if_,
    else_ctx_manager as else_,
)
from cusimt.mlir.dialect_exts import llvm
from cusimt.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
lower = registry.lower
from cusimt.lowering_utilities import (
    get_or_insert_function,
    lookup_callee_in_module,
    tensor_to_memref,
    memref_to_tensor,
    index_of,
    int_of,
)
from functools import singledispatch


def _dtype_format_string(dtype: ir.Type | type) -> str:
    match dtype:
        case ir.IntegerType():
            return "%d"
        case ir.FloatType():
            return "%f"
        case _ if dtype in (int, bool):
            return "%d"
        case _ if dtype is float:
            return "%f"
        case _ if dtype is str:
            return "%s"
        case _:
            raise ValueError(f"Unsupported type: {dtype}")


@singledispatch
def _lower_literal_print(value: int | float | bool | str) -> None:
    raise NotImplementedError(f"Unsupported type: {type(value)}")


@_lower_literal_print.register(int)
def _(value) -> None:
    mlir_value = arith.constant(T.i64(), value)
    return gpu.printf("%ld", mlir_value)


@_lower_literal_print.register(bool)
def _(value) -> None:
    return gpu.printf("True" if value else "False")


@_lower_literal_print.register(float)
def _(value) -> None:
    mlir_value = arith.constant(T.f64(), value)
    return gpu.printf("%f", mlir_value)


@_lower_literal_print.register(str)
def _(value) -> None:
    return gpu.printf(value)


def _memref_runtime_type_name(dtype: ir.Type) -> str:
    match dtype:
        case ir.IntegerType():
            return "I" + str(dtype.width)
        case ir.BF16Type():
            return "BF16"
        case ir.FloatType():
            return "F" + str(dtype.width)
        case _:
            raise ValueError(f"Unsupported type: {dtype}")


def _lower_memref_print(mlir_lower: MLIRLower, value: ir.Value) -> None:
    dtype = value.type.element_type
    value = tensor_to_memref(value)
    vt: ir.MemRefType = value.type
    if not vt.has_rank:
        raise TypeError("Cannot print unranked memref")
    rank = vt.rank
    c0 = index_of(0)
    c1 = index_of(1)
    gpu.printf(f"memref<shape=[")
    dims = []
    for iv in range(rank):
        dim = memref.dim(value, index_of(iv))
        dims.append(dim)
        gpu.printf("%d,", dim)

    # printing the data is probably not a good idea anyways (look at what pytorch tensor's
    # __str__ does) and it probably needs to be recursive and therefore defined in the
    # runtime...

    # gpu.printf("],data=[")
    # linear_mr = memref_to_tensor(value)
    # if rank > 1:
    #     mr_len = list(itertools.accumulate(dims, initial=1, func=operator.mul))
    #     linear_mr_dim = mr_len[-1]
    #     linear_mr_type = T.tensor(ir.RankedTensorType.get_dynamic_size(), dtype)
    #     linear_mr = tensor.collapse_shape(
    #         linear_mr_type, linear_mr, [list(range(rank))]
    #     )
    # else:
    #     linear_mr_dim = tensor.dim(linear_mr, index_of(0))

    # @scf.for__(c0, linear_mr_dim, c1)
    # def for_op(iv: T.index()):
    #     value_at_index = tensor.extract(linear_mr, [iv])
    #     gpu.printf("%d,", value_at_index)

    gpu.printf("]>")


def _lower_variable_print(mlir_lower: MLIRLower, value: ir.Value) -> None:
    match value.type:
        case ir.IntegerType() as int_type if int_type.width == 1:
            # Boolean (i1) - conditionally print "True" or "False"
            with if_(value, results=[]) as if_op:
                gpu.printf("True")
                scf.yield_([])
            with else_(if_op):
                gpu.printf("False")
                scf.yield_([])
        case ir.IntegerType():
            return gpu.printf(_dtype_format_string(value.type), value)
        case ir.BF16Type():
            # Convert bf16 to f32 for printing
            f32_val = arith.extf(T.f32(), value)
            return gpu.printf("%f", f32_val)
        case ir.F16Type():
            # Convert f16 to f32 for printing
            f32_val = arith.extf(T.f32(), value)
            return gpu.printf("%f", f32_val)
        case ir.FloatType():
            return gpu.printf(_dtype_format_string(value.type), value)
        case ir.MemRefType() | ir.RankedTensorType():
            return _lower_memref_print(mlir_lower, value)
        case _ if str(value.type).startswith("!llvm.struct"):
            _lower_string_struct_print(value)
        case _:
            raise ValueError(f"Unsupported type: {value.type}")


def _lower_string_struct_print(value: ir.Value) -> None:
    """Print a unicode_type string struct atomically via ``%s``.

    The struct layout is ``(ptr, i64, i32, i32, i64, ptr, ptr)`` where
    field 0 is the data pointer (to a null-terminated byte array).

    gpu.printf only accepts integer/float operands, so we pass the pointer
    as an i64 and rely on CUDA's printf ``%s`` implementation to interpret
    the integer as a device pointer.
    """
    data_ptr = llvm.extractvalue(llvm.PointerType.get(), value, [0])
    ptr_as_i64 = llvm.ptrtoint(res=T.i64(), arg=data_ptr)
    gpu.printf("%s", ptr_as_i64)


def _lower_tuple_print(mlir_lower: MLIRLower, values: list, sep: str = " ") -> None:
    """Print a tuple in format (v1, v2, v3) or (v1,) for single-element."""
    gpu.printf("(")
    for i, val in enumerate(values):
        if i > 0:
            gpu.printf(", ")
        _lower_print_arg(mlir_lower, val)
    if len(values) == 1:
        gpu.printf(",")
    gpu.printf(")")


def _lower_dim3_print(dim3_obj) -> None:
    """Print a Dim3 in format (x, y, z)."""
    from cusimt import cuda as cusimt_cuda

    # Determine which dim3 object this is
    if dim3_obj == cusimt_cuda.threadIdx:
        x_val = gpu.thread_id(gpu.Dimension.x)
        y_val = gpu.thread_id(gpu.Dimension.y)
        z_val = gpu.thread_id(gpu.Dimension.z)
    elif dim3_obj == cusimt_cuda.blockIdx:
        x_val = gpu.block_id(gpu.Dimension.x)
        y_val = gpu.block_id(gpu.Dimension.y)
        z_val = gpu.block_id(gpu.Dimension.z)
    elif dim3_obj == cusimt_cuda.blockDim:
        x_val = gpu.block_dim(gpu.Dimension.x)
        y_val = gpu.block_dim(gpu.Dimension.y)
        z_val = gpu.block_dim(gpu.Dimension.z)
    elif dim3_obj == cusimt_cuda.gridDim:
        x_val = gpu.grid_dim(gpu.Dimension.x)
        y_val = gpu.grid_dim(gpu.Dimension.y)
        z_val = gpu.grid_dim(gpu.Dimension.z)
    else:
        raise ValueError(f"Unknown Dim3 object: {dim3_obj}")

    gpu.printf("(")
    gpu.printf("%d", x_val)
    gpu.printf(", ")
    gpu.printf("%d", y_val)
    gpu.printf(", ")
    gpu.printf("%d", z_val)
    gpu.printf(")")


def _is_dim3_object(arg) -> bool:
    """Check if arg is a Dim3 object (threadIdx, blockIdx, etc.)."""
    from cusimt import cuda as cusimt_cuda

    return arg in (
        cusimt_cuda.threadIdx,
        cusimt_cuda.blockIdx,
        cusimt_cuda.blockDim,
        cusimt_cuda.gridDim,
    )


def _lower_print_arg(mlir_lower: MLIRLower, arg) -> None:
    """Lower a single print argument."""
    if _is_dim3_object(arg):
        _lower_dim3_print(arg)
    elif isinstance(arg, bool):
        _lower_literal_print(arg)
    elif isinstance(arg, (int, float, str)):
        _lower_literal_print(arg)
    elif isinstance(arg, (list, tuple)):
        _lower_tuple_print(mlir_lower, arg)
    elif isinstance(arg, ir.Value):
        _lower_variable_print(mlir_lower, arg)
    else:
        raise ValueError(f"Unsupported print argument type: {type(arg)}")


def _get_format_and_values(mlir_lower: MLIRLower, arg) -> tuple[str, list[ir.Value]]:
    """Get format string piece and values for a single argument.

    Returns (format_string, [values]) where values are ir.Values to pass to printf.
    For complex types that can't be handled in a single printf, returns (None, [])
    to indicate fallback to separate printf calls is needed.
    """
    if _is_dim3_object(arg):
        from cusimt import cuda as cusimt_cuda

        if arg == cusimt_cuda.threadIdx:
            x = gpu.thread_id(gpu.Dimension.x)
            y = gpu.thread_id(gpu.Dimension.y)
            z = gpu.thread_id(gpu.Dimension.z)
        elif arg == cusimt_cuda.blockIdx:
            x = gpu.block_id(gpu.Dimension.x)
            y = gpu.block_id(gpu.Dimension.y)
            z = gpu.block_id(gpu.Dimension.z)
        elif arg == cusimt_cuda.blockDim:
            x = gpu.block_dim(gpu.Dimension.x)
            y = gpu.block_dim(gpu.Dimension.y)
            z = gpu.block_dim(gpu.Dimension.z)
        elif arg == cusimt_cuda.gridDim:
            x = gpu.grid_dim(gpu.Dimension.x)
            y = gpu.grid_dim(gpu.Dimension.y)
            z = gpu.grid_dim(gpu.Dimension.z)
        return "(%lld, %lld, %lld)", [x, y, z]
    elif isinstance(arg, bool):
        return ("True" if arg else "False"), []
    elif isinstance(arg, int):
        return "%lld", [arith.constant(T.i64(), arg)]
    elif isinstance(arg, float):
        return "%f", [arith.constant(T.f64(), arg)]
    elif isinstance(arg, str):
        return arg, []
    elif isinstance(arg, (list, tuple)):
        # Build tuple format: (v1, v2, v3) or (v1,)
        fmt_parts = []
        values = []
        for i, val in enumerate(arg):
            if i > 0:
                fmt_parts.append(", ")
            sub_fmt, sub_vals = _get_format_and_values(mlir_lower, val)
            if sub_fmt is None:
                return None, []  # Can't handle in single printf
            fmt_parts.append(sub_fmt)
            values.extend(sub_vals)
        if len(arg) == 1:
            return "(" + "".join(fmt_parts) + ",)", values
        return "(" + "".join(fmt_parts) + ")", values
    elif isinstance(arg, ir.Value):
        match arg.type:
            case ir.IntegerType() as int_type if int_type.width == 1:
                return None, []
            case ir.IntegerType():
                return "%lld", [arg]
            case ir.BF16Type():
                return "%f", [arith.extf(T.f32(), arg)]
            case ir.F16Type():
                return "%f", [arith.extf(T.f32(), arg)]
            case ir.FloatType():
                return "%f", [arg]
            case _ if str(arg.type).startswith("!llvm.struct"):
                data_ptr = llvm.extractvalue(llvm.PointerType.get(), arg, [0])
                return "%s", [llvm.ptrtoint(res=T.i64(), arg=data_ptr)]
            case ir.MemRefType() | ir.RankedTensorType():
                return None, []
            case _:
                return None, []
    return None, []


@lower(print, types.VarArg(types.Any))
def lower_print(
    mlir_lower: MLIRLower, target, args, kwargs: list[tuple[str, ir.Value]]
):
    def _kwarg_str(kw_list, name, default):
        """Extract a string keyword arg from its literal type."""
        for k, v in kw_list:
            if k == name:
                ty = mlir_lower.get_numba_type(v.name)
                if isinstance(ty, types.StringLiteral):
                    return ty.literal_value
                loaded = mlir_lower.load_var(v)
                return loaded if isinstance(loaded, str) else default
        return default

    end = _kwarg_str(kwargs, "end", "\n")
    sep = _kwarg_str(kwargs, "sep", " ")
    remaining_kwargs = {
        k: mlir_lower.load_var(v) for k, v in kwargs if k not in ("end", "sep")
    }
    args = [mlir_lower.load_var(arg) for arg in args]

    if remaining_kwargs:
        raise ValueError(f"Unsupported keyword arguments: {remaining_kwargs}")

    # Try to build a single format string for all arguments
    fmt_parts = []
    all_values = []
    can_combine = True

    for i, arg in enumerate(args):
        if i > 0:
            fmt_parts.append(sep)
        fmt, values = _get_format_and_values(mlir_lower, arg)
        if fmt is None:
            can_combine = False
            break
        fmt_parts.append(fmt)
        all_values.extend(values)

    if can_combine and args:
        # Emit single printf with combined format string
        format_str = "".join(fmt_parts) + end
        gpu.printf(format_str, *all_values)
    elif not args:
        # Empty print - just print newline
        gpu.printf(end)
    else:
        # Fall back to separate printf calls for complex cases
        for i, arg in enumerate(args):
            if i > 0:
                gpu.printf(sep)
            _lower_print_arg(mlir_lower, arg)
        gpu.printf(end)
