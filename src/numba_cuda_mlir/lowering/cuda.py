# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from itertools import islice

    def batched(iterable, n):
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


from numba_cuda_mlir._mlir.extras.meta import region_op
import operator
from numba_cuda_mlir.cuda.misc.special import literal_unroll, literally
from numba_cuda_mlir.numba_cuda.misc.special import (
    literal_unroll as cuda_literal_unroll,
)
from numba_cuda_mlir.errors import InternalCompilerError
from numba_cuda_mlir.lowering_utilities import (
    coerce_to_shape_tuple,
    constant,
    try_extract_constant,
    int_of,
    index_of,
    convert,
    f32_of,
    i32_of,
    memref_to_llvm_ptr,
    storage_itemsize_bytes,
    storage_bitwidth,
    memref_llvm_address_space,
)
from numba_cuda_mlir.lowering_utilities.type_conversions import (
    to_mlir_type,
    to_numba_type,
    inline_ptx_type_constraint_to_numba_type,
    inline_ptx_type_constraint_to_mlir_type,
)
from numba_cuda_mlir.descriptor import MLIRTargetContext
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir._mlir.dialects import arith, builtin, func, memref, gpu, nvvm, llvm
from numba_cuda_mlir._mlir.extras import types as T
import numba_cuda_mlir._mlir.ir as ir
from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
lower = registry.lower
lower_getattr = registry.lower_getattr
lower_getattr_generic = registry.lower_getattr_generic
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import fp16
import numba_cuda_mlir.cuda
from numba_cuda_mlir import types
import numba_cuda_mlir.numba_cuda.core.ir as numba_ir
import numba_cuda_mlir.lowering_utilities.type_conversions as type_conversions
from numba_cuda_mlir.numba_cuda.types.ext_types import Dim3, GridGroup as GridGroupClass
from numba_cuda_mlir.numba_cuda.cg import this_grid
from numba_cuda_mlir.numba_cuda.extending import (
    overload,
    overload_attribute,
    intrinsic,
    overload_method,
)
from typing import Any
from numba_cuda_mlir.logging import trace
import numpy as np
from numba_cuda_mlir.numba_cuda.typing.templates import ConcreteTemplate
from numba_cuda_mlir.numba_cuda import stubs as cuda_stubs


def _lower_cfarray(builder: MLIRLower, target, args, kwargs, name, layout):
    if kwargs and [name for name, _ in kwargs] != ["dtype"]:
        raise TypeError(f"{name} only supports dtype as a keyword argument")

    ptr = builder.load_var(args[0])
    shape = builder.load_var(args[1])
    result_type = builder.get_numba_type(target)
    rank = result_type.ndim

    if isinstance(shape, tuple):
        sizes = list(shape)
    else:
        sizes = [shape]

    if len(sizes) != rank:
        raise TypeError(f"shape rank {len(sizes)} does not match array rank {rank}")

    sizes_i64 = [convert(size, T.i64()) for size in sizes]

    strides_i64 = [None] * rank
    stride = constant(1, T.i64())
    indices = range(rank - 1, -1, -1) if layout == "C" else range(rank)
    for i in indices:
        strides_i64[i] = stride
        stride = arith.muli(stride, sizes_i64[i])

    result_mlir_type = builder.get_mlir_type(result_type)
    i64 = T.i64()

    if rank > 0:
        struct_type = ir.Type.parse(
            f"!llvm.struct<(ptr, ptr, i64, array<{rank} x i64>, array<{rank} x i64>)>"
        )
    else:
        struct_type = ir.Type.parse("!llvm.struct<(ptr, ptr, i64)>")

    desc = llvm.UndefOp(struct_type).result
    zero = constant(0, i64)
    ins = lambda d, v, *p: llvm.insertvalue(
        container=d, value=v, position=ir.DenseI64ArrayAttr.get(list(p))
    )

    desc = ins(desc, ptr, 0)
    desc = ins(desc, ptr, 1)
    desc = ins(desc, zero, 2)
    for i, size in enumerate(sizes_i64):
        desc = ins(desc, size, 3, i)
    for i, stride in enumerate(strides_i64):
        desc = ins(desc, stride, 4, i)

    builder.store_var(target, builtin.unrealized_conversion_cast([result_mlir_type], [desc]))


@lower(cuda.carray, types.CPointer, types.Integer)
@lower(cuda.carray, types.CPointer, types.BaseTuple)
@lower(cuda.carray, types.CPointer, types.Integer, types.Any)
@lower(cuda.carray, types.CPointer, types.BaseTuple, types.Any)
@lower(cuda.carray, types.voidptr, types.Integer, types.Any)
@lower(cuda.carray, types.voidptr, types.BaseTuple, types.Any)
def lower_carray(builder: MLIRLower, target, args, kwargs):
    _lower_cfarray(builder, target, args, kwargs, "carray", "C")


@lower(cuda.farray, types.CPointer, types.Integer)
@lower(cuda.farray, types.CPointer, types.BaseTuple)
@lower(cuda.farray, types.CPointer, types.Integer, types.Any)
@lower(cuda.farray, types.CPointer, types.BaseTuple, types.Any)
@lower(cuda.farray, types.voidptr, types.Integer, types.Any)
@lower(cuda.farray, types.voidptr, types.BaseTuple, types.Any)
def lower_farray(builder: MLIRLower, target, args, kwargs):
    _lower_cfarray(builder, target, args, kwargs, "farray", "F")


@lower(cuda_stubs.nanosleep, types.Number)
def cuda_nanosleep(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    ticks = lower.load_var(args[0])
    ticks = convert(ticks, T.i32())
    nvvm.nanosleep(ticks)
    lower.store_var(target, None)


@lower(numba_cuda_mlir.cuda.selp, types.Integer, types.Integer, types.Integer)
@lower(numba_cuda_mlir.cuda.selp, types.Boolean, types.Integer, types.Integer)
@lower(numba_cuda_mlir.cuda.selp, types.Integer, types.Float, types.Float)
@lower(numba_cuda_mlir.cuda.selp, types.Boolean, types.Float, types.Float)
def cuda_selp(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    cond = lower.load_var(args[0])
    true_val = lower.load_var(args[1])
    false_val = lower.load_var(args[2])
    # Convert condition to i1 (bool) if needed
    if cond.type != T.bool():
        zero = arith.constant(result=cond.type, value=0)
        cond = arith.cmpi(arith.CmpIPredicate.ne, cond, zero)
    # Ensure true_val and false_val have same type
    if true_val.type != false_val.type:
        target_type = true_val.type
        false_val = convert(false_val, target_type)
    result = arith.select(cond, true_val, false_val)
    lower.store_var(target, result)


def _lower_warpsize(context, lower: MLIRLower, target, cuda_mod):
    # Warp size is always 32 on NVIDIA GPUs, but use the NVVM intrinsic for correctness
    result = nvvm.read_ptx_sreg_warpsize()
    lower.store_var(target, result)


def _lower_laneid(context, lower: MLIRLower, target, cuda_mod):
    # Lane ID is 0-31 within a warp
    result = nvvm.read_ptx_sreg_laneid()
    lower.store_var(target, result)


# Register warpsize and laneid for the numba_cuda_mlir.cuda module.
# The name-based fallback in MLIRTargetContext._find_module_getattr_by_name()
# will match this lowering for equivalent cuda modules when exact type matching fails.
@lower_getattr(types.Module(numba_cuda_mlir.cuda), "warpsize")
def lower_cuda_warpsize(context, lower, target, cuda_mod):
    return _lower_warpsize(context, lower, target, cuda_mod)


@lower_getattr(types.Module(numba_cuda_mlir.cuda), "laneid")
def lower_cuda_laneid(context, lower, target, cuda_mod):
    return _lower_laneid(context, lower, target, cuda_mod)


@lower(numba_cuda_mlir.cuda.brev, types.Integer)
def cuda_brev(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    value = lower.load_var(args[0])
    # Use the Numba type's bitwidth to determine 32 vs 64-bit operation
    numba_type = lower.get_numba_type(args[0])
    if numba_type.bitwidth > 32:
        value = convert(value, T.i64())
    else:
        value = convert(value, T.i32())
    result = llvm.intr_bitreverse(value)
    lower.store_var(target, result)


@lower(numba_cuda_mlir.cuda.clz, types.Integer)
def cuda_clz(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    value = lower.load_var(args[0])
    # Use the Numba type's bitwidth to determine 32 vs 64-bit operation
    numba_type = lower.get_numba_type(args[0])
    if numba_type.bitwidth > 32:
        value = convert(value, T.i64())
    else:
        value = convert(value, T.i32())
    # is_zero_poison=False means clz(0) returns bit width
    result = llvm.intr_ctlz(value, False)
    result = convert(result, T.i32())
    lower.store_var(target, result)


@lower(numba_cuda_mlir.cuda.popc, types.Integer)
def cuda_popc(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    value = lower.load_var(args[0])
    # Use the Numba type's bitwidth to determine 32 vs 64-bit operation
    numba_type = lower.get_numba_type(args[0])
    if numba_type.bitwidth > 32:
        value = convert(value, T.i64())
    else:
        value = convert(value, T.i32())
    result = llvm.intr_ctpop(value)
    result = convert(result, T.i32())
    lower.store_var(target, result)


@lower(numba_cuda_mlir.cuda.ffs, types.Integer)
def cuda_ffs(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    value = lower.load_var(args[0])
    # Use the Numba type's bitwidth to determine 32 vs 64-bit operation
    numba_type = lower.get_numba_type(args[0])
    if numba_type.bitwidth > 32:
        int_type = T.i64()
    else:
        int_type = T.i32()
    value = convert(value, int_type)
    # ffs returns 1-indexed position of least significant set bit, or 0 if input is 0
    # cttz returns count of trailing zeros (0 for odd numbers)
    # ffs(x) = cttz(x) + 1 for x != 0, ffs(0) = 0
    cttz = llvm.intr_cttz(value, False)  # is_zero_poison=False
    one = arith.constant(result=int_type, value=1)
    zero = arith.constant(result=int_type, value=0)
    ffs_nonzero = arith.addi(cttz, one)
    is_zero = arith.cmpi(arith.CmpIPredicate.eq, value, zero)
    result = arith.select(is_zero, zero, ffs_nonzero)
    result = convert(result, T.i32())
    lower.store_var(target, result)


@lower(numba_cuda_mlir.cuda.syncthreads)
def cuda_syncthreads(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    gpu.barrier()
    lower.store_var(target, None)


def _ensure_cudadevrt_linked(lower: MLIRLower):
    if not lower.metadata.get("_cudadevrt_linked"):
        from numba_cuda_mlir.numba_cuda.cudadrv.libs import get_cudalib

        lower.link_external_item(get_cudalib("cudadevrt", static=True))
        lower.metadata["_cudadevrt_linked"] = True


@lower(this_grid)
def lower_cg_this_grid(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    from numba_cuda_mlir.lowering_utilities import get_or_insert_function

    fn_type = ir.FunctionType.get(results=[T.i64()], inputs=[T.i32()])
    callee = get_or_insert_function("cudaCGGetIntrinsicHandle", fn_type, lower.mlir_gpu_module)
    one = arith.constant(result=T.i32(), value=1)
    result = func.call(result=[T.i64()], callee=callee.name.value, operands_=[one])
    lower.metadata["use_cooperative"] = True
    _ensure_cudadevrt_linked(lower)
    lower.store_var(target, result)


def _grid_group_sync_cg(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    from numba_cuda_mlir.lowering_utilities import get_or_insert_function

    group = lower.load_var(args[0])
    fn_type = ir.FunctionType.get(results=[T.i32()], inputs=[T.i64(), T.i32()])
    callee = get_or_insert_function("cudaCGSynchronize", fn_type, lower.mlir_gpu_module)
    zero = arith.constant(result=T.i32(), value=0)
    result = func.call(result=[T.i32()], callee=callee.name.value, operands_=[group, zero])
    lower.metadata["use_cooperative"] = True
    _ensure_cudadevrt_linked(lower)
    lower.store_var(target, result)


@lower_getattr(GridGroupClass, "sync")
def lower_grid_group_sync_getattr(context, lower, target, grid_group_var):
    from numba_cuda_mlir.lowering_utilities import DeferredMethodCall

    lower.store_var(target, DeferredMethodCall(grid_group_var, _grid_group_sync_cg))


def _validate_alignment(alignment: int):
    """
    Validates alignment value. Raises ValueError if invalid.
    Valid alignment must be: positive, power of 2, multiple of pointer size (8).
    """
    import struct

    if alignment is None:
        return
    if not isinstance(alignment, int):
        raise ValueError("Alignment must be an integer")
    if alignment <= 0:
        raise ValueError("Alignment must be positive")
    if (alignment & (alignment - 1)) != 0:
        raise ValueError("Alignment must be a power of 2")
    pointer_size = struct.calcsize("P")
    if (alignment % pointer_size) != 0:
        raise ValueError(f"Alignment must be a multiple of {pointer_size}")


def _extract_shape_and_dtype(shape, dtype, alignment=None, alignas=None, **kwargs):
    assert not kwargs, f"got unexpected keyword arguments: {kwargs}"
    # Support both 'alignment' (numba-cuda) and 'alignas' (numba_cuda_mlir) keywords
    align_val = alignment if alignment is not None else alignas
    if align_val is None:
        align_val = 8
    return shape, dtype, align_val


def _resolve_dtype(lower, dtype_var):
    """Resolve dtype from a Var, handling StringLiteral (e.g. dtype="int32")."""
    if isinstance(dtype_var, numba_ir.Var):
        dtype_numba = lower.get_numba_type(dtype_var.name)
        if isinstance(dtype_numba, types.StringLiteral):
            return np.dtype(dtype_numba.literal_value)
        if isinstance(dtype_numba, types.DTypeSpec):
            return dtype_numba
        return lower.load_var(dtype_var)
    return dtype_var


def _resolve_numba_dtype(lower, dtype_var):
    dtype = _resolve_dtype(lower, dtype_var)
    if isinstance(dtype, np.dtype):
        return to_numba_type(dtype)
    if isinstance(dtype, type) and getattr(dtype, "__module__", None) == "numpy":
        return to_numba_type(np.dtype(dtype))
    if isinstance(dtype, types.DTypeSpec) and hasattr(dtype, "dtype"):
        return dtype.dtype
    return dtype


shmem_id = 0


def cuda_static_shared_memory(lower: MLIRLower, target, static_shape, dtype, alignas):
    global shmem_id
    shape = tuple(static_shape)
    dtype = lower.get_storage_type(_resolve_numba_dtype(lower, dtype))
    mspace = ir.Attribute.parse("#gpu.address_space<workgroup>")
    ty = T.memref(*shape, element_type=dtype, memory_space=mspace)
    gpu_module = lower.mlir_gpu_module
    static_shared_memory_name = f"static_shared_memory_{shmem_id}"
    with ir.InsertionPoint(gpu_module.bodyRegion.blocks[0]):
        memref.global_(
            static_shared_memory_name,
            ty,
            sym_visibility="private",
            alignment=alignas,
        )
        shmem_id += 1
    mr = memref.get_global(ty, static_shared_memory_name)
    lower.store_var(target, mr)


@lower(cuda.shared.array, types.Number)
@lower(cuda.shared.array, types.Number, types.DTypeSpec)
@lower(cuda.shared.array, types.Number, types.StringLiteral)
@lower(cuda.shared.array, types.Number, types.DTypeSpec, types.Number)
@lower(cuda.shared.array, types.Number, types.StringLiteral, types.Number)
@lower(cuda.shared.array, types.Number, types.DTypeSpec, types.IntegerLiteral)
@lower(cuda.shared.array, types.Number, types.StringLiteral, types.IntegerLiteral)
@lower(cuda.shared.array, types.Number, types.DTypeSpec, types.NoneType)
@lower(cuda.shared.array, types.Number, types.StringLiteral, types.NoneType)
@lower(cuda.shared.array, types.UniTuple)
@lower(cuda.shared.array, types.UniTuple, types.DTypeSpec)
@lower(cuda.shared.array, types.UniTuple, types.StringLiteral)
@lower(cuda.shared.array, types.UniTuple, types.DTypeSpec, types.Number)
@lower(cuda.shared.array, types.UniTuple, types.StringLiteral, types.Number)
@lower(cuda.shared.array, types.UniTuple, types.DTypeSpec, types.IntegerLiteral)
@lower(cuda.shared.array, types.UniTuple, types.StringLiteral, types.IntegerLiteral)
@lower(cuda.shared.array, types.UniTuple, types.DTypeSpec, types.NoneType)
@lower(cuda.shared.array, types.UniTuple, types.StringLiteral, types.NoneType)
@lower(cuda.shared.array, types.Tuple)
@lower(cuda.shared.array, types.Tuple, types.DTypeSpec)
@lower(cuda.shared.array, types.Tuple, types.StringLiteral)
@lower(cuda.shared.array, types.Tuple, types.DTypeSpec, types.Number)
@lower(cuda.shared.array, types.Tuple, types.StringLiteral, types.Number)
@lower(cuda.shared.array, types.Tuple, types.DTypeSpec, types.IntegerLiteral)
@lower(cuda.shared.array, types.Tuple, types.StringLiteral, types.IntegerLiteral)
@lower(cuda.shared.array, types.Tuple, types.DTypeSpec, types.NoneType)
@lower(cuda.shared.array, types.Tuple, types.StringLiteral, types.NoneType)
def cuda_shared_memory(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    shape, dtype, alignas = _extract_shape_and_dtype(*args, **dict(kwargs))
    if isinstance(alignas, numba_ir.Var):
        # Check if the variable has NoneType - if so, use default alignment
        alignas_type = lower.get_numba_type(alignas.name)
        if isinstance(alignas_type, types.NoneType):
            alignas = None  # will be set to default after validation
        else:
            alignas = lower.load_var(alignas)
            alignas = try_extract_constant(alignas)
    else:
        alignas = try_extract_constant(alignas)
    # Validate alignment (raises ValueError for invalid values)
    _validate_alignment(alignas)
    if alignas is None:
        alignas = 8  # default alignment when None
    shape_op: tuple[ir.Value | int, ...] | ir.Value | int = lower.load_var(shape)

    # Wrap single values (int or ir.Value) in a tuple for 1D arrays
    if isinstance(shape_op, (ir.Value, int)):
        shape_op = (shape_op,)

    def _is_static_dim(x):
        match x:
            case int():
                return x
            case ir.Value() if isinstance(x.owner, ir.Block):
                return None  # Block arguments are dynamic
            case ir.Value():
                return _is_static_dim(x.owner.opview)
            case arith.ConstantOp():
                return x.value.value
            case _:
                return None

    static_shape = [_is_static_dim(x) for x in shape_op]

    is_dynamic_shared_shape = len(static_shape) == 1 and static_shape[0] == 0
    if all([x is not None for x in static_shape]) and not is_dynamic_shared_shape:
        return cuda_static_shared_memory(lower, target, static_shape, dtype, alignas)

    shape = coerce_to_shape_tuple(shape_op)

    np_dtype = _resolve_numba_dtype(lower, dtype)
    dtype = lower.get_storage_type(np_dtype)
    mr_type = ir.MemRefType.get(
        shape=[ir.MemRefType.get_dynamic_size() for _ in shape],
        element_type=dtype,
        memory_space=lower._get_shared_address_space(),
    )
    if is_dynamic_shared_shape:
        array = lower._request_dynamic_shared_memory(mr_type)
    else:
        array = lower._request_shared_memory(shape, mr_type)
    if alignas != 8:
        array = lower._mark_dynamic_shared_memory_alias(
            memref.assume_alignment(array, alignas), array
        )
    lower.store_var(target, array)


@lower(cuda.local_array, types.Tuple, types.DTypeSpec, types.Number)
@lower(cuda.local_array, types.Number, types.DTypeSpec, types.Number)
@lower(cuda.local_array, types.Tuple, types.DTypeSpec)
@lower(cuda.local_array, types.Number, types.DTypeSpec)
@lower(cuda.local_array, types.Tuple, types.StringLiteral)
@lower(cuda.local_array, types.Number, types.StringLiteral)
@lower(cuda.local_array, types.Tuple, types.DTypeSpec, types.NoneType)
@lower(cuda.local_array, types.Number, types.DTypeSpec, types.NoneType)
@lower(cuda.local_array, types.Tuple, types.StringLiteral, types.NoneType)
@lower(cuda.local_array, types.Number, types.StringLiteral, types.NoneType)
@lower(cuda.local_array, types.UniTuple, types.DTypeSpec)
@lower(cuda.local_array, types.UniTuple, types.StringLiteral)
@lower(cuda.local_array, types.UniTuple, types.DTypeSpec, types.Number)
@lower(cuda.local_array, types.UniTuple, types.DTypeSpec, types.NoneType)
@lower(cuda.local_array, types.UniTuple, types.StringLiteral, types.NoneType)
@lower(cuda.local_array, types.Tuple, types.DTypeSpec, types.IntegerLiteral)
@lower(cuda.local_array, types.Number, types.DTypeSpec, types.IntegerLiteral)
@lower(cuda.local_array, types.Tuple, types.StringLiteral, types.IntegerLiteral)
@lower(cuda.local_array, types.Number, types.StringLiteral, types.IntegerLiteral)
@lower(cuda.local_array, types.UniTuple, types.DTypeSpec, types.IntegerLiteral)
@lower(cuda.local_array, types.UniTuple, types.StringLiteral, types.IntegerLiteral)
def cuda_local_array(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    shape, dtype, alignas = _extract_shape_and_dtype(*args, **dict(kwargs))
    shape_op: tuple[ir.Value | int, ...] | ir.Value | int = lower.load_var(shape)
    if isinstance(alignas, numba_ir.Var):
        # Check if the variable has NoneType - if so, use default alignment
        alignas_type = lower.get_numba_type(alignas.name)
        if isinstance(alignas_type, types.NoneType):
            alignas = None  # will be set to default after validation
        else:
            alignas = lower.load_var(alignas)
            alignas = try_extract_constant(alignas)
    else:
        alignas = try_extract_constant(alignas)
    # Validate alignment (raises ValueError for invalid values)
    _validate_alignment(alignas)
    if alignas is None:
        alignas = 8  # default alignment

    # Wrap single values (int or ir.Value) in a tuple for 1D arrays
    if isinstance(shape_op, (ir.Value, int)):
        shape_op = (shape_op,)

    # Try to extract static shape values
    static_shape = []
    for dim in shape_op:
        const_val = try_extract_constant(dim)
        if const_val is not None:
            static_shape.append(int(const_val))
        else:
            static_shape = None
            break

    np_dtype = _resolve_numba_dtype(lower, dtype)
    mlir_dtype = lower.get_storage_type(np_dtype)

    if static_shape is not None:
        # Static shape - use static memref allocation
        with lower.alloca_insertion_point():
            mr_type = ir.MemRefType.get(
                shape=static_shape,
                element_type=mlir_dtype,
            )
            # Pass alignment attribute if non-default
            if alignas != 8:
                mr = memref.alloca(mr_type, [], [], alignment=alignas)
            else:
                mr = memref.alloca(mr_type, [], [])
    else:
        # Dynamic shape - need to compute shape in entry block
        shape = coerce_to_shape_tuple(shape_op)
        with lower.alloca_insertion_point():
            mr_type = ir.MemRefType.get(
                shape=[ir.MemRefType.get_dynamic_size() for _ in shape],
                element_type=mlir_dtype,
            )
            # Pass alignment attribute if non-default
            if alignas != 8:
                mr = memref.alloca(
                    mr_type,
                    dynamic_sizes=shape,
                    symbol_operands=[],
                    alignment=alignas,
                )
            else:
                mr = memref.alloca(
                    mr_type,
                    dynamic_sizes=shape,
                    symbol_operands=[],
                )
    lower.store_var(target, mr)


@lower(cuda.const.array_like, types.Array)
def cuda_const_array_like(lower: MLIRLower, target, args: list[Any], kwargs: list[tuple[str, Any]]):
    # const.array_like is essentially a no-op at lowering time - the array
    # already exists and will be passed as a kernel argument. The constant
    # memory placement is handled by the CUDA driver when the data is copied.
    arr = args[0]
    lower.store_var(target, lower.load_var(arr))


@lower_getattr(types.Array, "dtype")
def lower_array_dtype(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    trace()
    array_type = mlir_lower.get_numba_type(array.name)
    mlir_lower.store_var(target, array_type.dtype)


@lower_getattr(types.Array, "shape")
def lower_shape(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    array = mlir_lower.load_var(array)
    array_type = array.type
    result = tuple(
        memref.dim(source=array, index=arith.constant(result=T.index(), value=i))
        for i in range(array_type.rank)
    )
    mlir_lower.store_var(target, result)


@lower_getattr(types.Array, "strides")
def lower_strides(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    from numba_cuda_mlir.lowering_utilities import int_of, index_of

    array = mlir_lower.load_var(array)
    array_type = array.type
    rank = array_type.rank
    array_numba_type = mlir_lower.get_numba_type(array.name)

    # Get element size in bytes
    element_size = storage_itemsize_bytes(array_numba_type)

    # Get dimensions - handle both memref and tensor types
    if isinstance(array_type, ir.MemRefType):
        dims = [
            memref.dim(source=array, index=arith.constant(result=T.index(), value=i))
            for i in range(rank)
        ]
    elif isinstance(array_type, ir.RankedTensorType):
        dims = [
            tensor.dim(source=array, index=arith.constant(result=T.index(), value=i))
            for i in range(rank)
        ]
    else:
        raise NotImplementedError(f"strides not implemented for {array_type}")

    # Compute strides (C-contiguous order)
    # stride[rank-1] = element_size
    # stride[i] = stride[i+1] * dim[i+1]
    strides = [None] * rank
    strides[-1] = index_of(element_size)
    for i in range(rank - 2, -1, -1):
        strides[i] = arith.muli(strides[i + 1], dims[i + 1])

    mlir_lower.store_var(target, tuple(strides))


def _dim3_attr_to_dimension(attr: str):
    match attr:
        case "x":
            return gpu.Dimension.x
        case "y":
            return gpu.Dimension.y
        case "z":
            return gpu.Dimension.z
        case _:
            raise ValueError(f"Invalid attribute for Dim3: {attr}")


def _get_dim3_attribute(
    which_one,
    dimension: gpu.Dimension,
) -> ir.Value:
    match which_one:
        case cuda.threadIdx:
            return gpu.thread_id(dimension)
        case cuda.blockIdx:
            return gpu.block_id(dimension)
        case cuda.blockDim:
            return gpu.block_dim(dimension)
        case cuda.gridDim:
            return gpu.grid_dim(dimension)
        case _:
            raise ValueError(f"Invalid attribute for Dim3: {which_one}, attribute: {dimension}")


@lower_getattr_generic(Dim3)
def lower_dim3_getattr(
    _: MLIRTargetContext,
    builder: MLIRLower,
    target: numba_ir.Var,
    value: numba_ir.Var,
    attr: str,
):
    dimension = _dim3_attr_to_dimension(attr)
    attr_value = builder.load_var(value)
    assert attr_value in (
        cuda.threadIdx,
        cuda.blockIdx,
        cuda.blockDim,
        cuda.gridDim,
    ), f"Expected cuda Dim3 instance, got {attr_value}"
    res = _get_dim3_attribute(attr_value, dimension)
    to_type = builder.get_numba_type(target.name)
    to_type = builder.get_mlir_type(to_type)
    res = builder.mlir_convert(res, to_type)
    builder.store_var(target, res)


def _compute_global_tid(dimension, to_type: ir.Type):
    """
    Helper function to compute global thread ID for a given dimension
    """
    tid = gpu.thread_id(dimension=dimension)
    bdim = gpu.block_dim(dimension=dimension)
    bid = gpu.block_id(dimension=dimension)
    res = arith.muli(lhs=bid, rhs=bdim)
    res = arith.addi(lhs=res, rhs=tid)
    res = convert(res, to_type)
    trace("dimension=%s res=%s (%s)", dimension, res, type(res))
    return res


@lower(cuda.grid, types.Number)
def lower_cuda_grid(
    lower: MLIRLower,
    target: numba_ir.Var,
    args: list[Any],
    kwargs: list[tuple[str, Any]],
):
    target_type = lower.get_numba_type(target.name)

    # Check if dimension value is constant
    arg_type = lower.get_numba_type(args[0].name)
    if not isinstance(arg_type, (types.IntegerLiteral, types.Literal)):
        # Runtime dimension value - not typically used, but we could support it
        raise NotImplementedError("cuda.grid() with runtime dimension value is not supported")

    # Constant dimension - compile-time branch
    dim_literal = arg_type.literal_value
    if dim_literal == 1:
        # Return single value for grid(1)
        result_type = lower.get_mlir_type(target_type)
        result = _compute_global_tid(gpu.Dimension.x, result_type)
        lower.store_var(target, result)
    elif dim_literal == 2:
        element_type = lower.get_mlir_type(target_type.dtype)
        result_x = _compute_global_tid(gpu.Dimension.x, element_type)
        result_y = _compute_global_tid(gpu.Dimension.y, element_type)
        lower.store_var(target, (result_x, result_y))
    elif dim_literal == 3:
        element_type = lower.get_mlir_type(target_type.dtype)
        result_x = _compute_global_tid(gpu.Dimension.x, element_type)
        result_y = _compute_global_tid(gpu.Dimension.y, element_type)
        result_z = _compute_global_tid(gpu.Dimension.z, element_type)
        lower.store_var(target, (result_x, result_y, result_z))
    else:
        raise ValueError(f"cuda.grid() only supports dimensions 1, 2, or 3, got {dim_literal}")


@lower(cuda.gridsize, types.Number)
def lower_cuda_gridsize(
    lower: MLIRLower,
    target: numba_ir.Var,
    args: list[Any],
    kwargs: list[tuple[str, Any]],
):
    assert len(args) == 1, "cuda.gridsize expects one positional argument"

    # Helper function to compute grid size for a given dimension
    # gridsize = gridDim * blockDim
    def _compute_gridsize(dimension, to_type: ir.Type):
        gdim = gpu.grid_dim(dimension=dimension)
        bdim = gpu.block_dim(dimension=dimension)
        res = arith.muli(lhs=gdim, rhs=bdim)
        return convert(res, to_type)

    # Check if dimension value is constant
    arg_type = lower.get_numba_type(args[0].name)
    if not isinstance(arg_type, (types.IntegerLiteral, types.Literal)):
        # Runtime dimension value - not typically used, but we could support it
        raise NotImplementedError("cuda.gridsize() with runtime dimension value is not supported")

    target_type = lower.get_numba_type(target.name)

    # Constant dimension - compile-time branch
    dim_literal = arg_type.literal_value
    if dim_literal == 1:
        mlir_type = lower.get_mlir_type(target_type)
        result = _compute_gridsize(gpu.Dimension.x, mlir_type)
        lower.store_var(target, result)
    elif dim_literal == 2:
        element_type = lower.get_mlir_type(target_type.dtype)
        result_x = _compute_gridsize(gpu.Dimension.x, element_type)
        result_y = _compute_gridsize(gpu.Dimension.y, element_type)
        lower.store_var(target, (result_x, result_y))
    elif dim_literal == 3:
        element_type = lower.get_mlir_type(target_type.dtype)
        result_x = _compute_gridsize(gpu.Dimension.x, element_type)
        result_y = _compute_gridsize(gpu.Dimension.y, element_type)
        result_z = _compute_gridsize(gpu.Dimension.z, element_type)
        lower.store_var(target, (result_x, result_y, result_z))
    else:
        raise ValueError(f"cuda.gridsize() only supports dimensions 1, 2, or 3, got {dim_literal}")


def _tuple_size_from_dimension(dim: types.Type):
    match dim:
        case types.IntegerLiteral(literal_value=1):
            return types.int64
        case types.IntegerLiteral(literal_value=2):
            return types.UniTuple(types.int64, 2)
        case types.IntegerLiteral(literal_value=3):
            return types.UniTuple(types.int64, 3)
        case _:
            raise ValueError(f"cuda.gridsize() only supports dimensions 1, 2, or 3, got {dim}")


def get_syncthreads_variant(reduction_op: nvvm.BarrierReduction):
    def cg_syncthreads_variant(builder, target, args, kwargs):
        pred = builder.load_var(args[0]) if len(args) else None
        pred = convert(pred, T.i32())
        res = nvvm.barrier(
            barrier_id=None,
            number_of_threads=None,
            reduction_op=reduction_op,
            reduction_predicate=pred,
        )
        builder.store_var(target, res)

    return cg_syncthreads_variant


@lower(numba_cuda_mlir.cuda.syncwarp)
@lower(numba_cuda_mlir.cuda.syncwarp, types.Number)
def cuda_syncwarp_cg(builder, target, args, kwargs):
    mask = 0xFFFFFFFF
    if args:
        mask = builder.load_var(args[0])
    mask = int_of(mask, T.i32())
    res = nvvm.bar_warp_sync(mask)
    builder.store_var(target, res)


def _shfl_sync_lowering(builder: MLIRLower, target, args, kwargs, shfl_kind):
    mask = builder.load_var(args[0])
    value = builder.load_var(args[1])
    offset = builder.load_var(args[2])

    # nvvm.shfl.sync only supports i32 or f32
    mask = int_of(mask, T.i32())
    offset = int_of(offset, T.i32())
    original_type = value.type
    value = f32_of(value) if isinstance(value.type, ir.FloatType) else i32_of(value)

    mask_and_clamp = constant(0 if shfl_kind == nvvm.ShflKind.up else 0x1F, T.i32())
    result = nvvm.shfl_sync(mask, value, offset, mask_and_clamp, shfl_kind)

    # Convert result back to original type if needed
    if isinstance(original_type, ir.FloatType) and original_type.width == 64:
        result = arith.extf(T.f64(), result)

    builder.store_var(target, result)


def register_shfl_sync_lowerings():
    from numba_cuda_mlir import cuda

    def make_shfl_lowering(shfl_kind):
        def lowering(builder, target, args, kwargs):
            _shfl_sync_lowering(builder, target, args, kwargs, shfl_kind)

        return lowering

    lower(cuda.shfl_sync, types.Integer, types.Number, types.Integer)(
        make_shfl_lowering(nvvm.ShflKind.idx)
    )
    lower(cuda.shfl_up_sync, types.Integer, types.Number, types.Integer)(
        make_shfl_lowering(nvvm.ShflKind.up)
    )
    lower(cuda.shfl_down_sync, types.Integer, types.Number, types.Integer)(
        make_shfl_lowering(nvvm.ShflKind.down)
    )
    lower(cuda.shfl_xor_sync, types.Integer, types.Number, types.Integer)(
        make_shfl_lowering(nvvm.ShflKind.bfly)
    )


register_shfl_sync_lowerings()


def register_syncthreads_variants():
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir._mlir.dialects import nvvm

    for intrin, reduction_op in [
        (cuda.syncthreads_and, nvvm.BarrierReduction.AND),
        (cuda.syncthreads_or, nvvm.BarrierReduction.OR),
        (cuda.syncthreads_count, nvvm.BarrierReduction.POPC),
    ]:
        lower(intrin, types.Integer)(get_syncthreads_variant(reduction_op))


register_syncthreads_variants()


def register_vote_sync_lowerings():
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir._mlir.dialects import nvvm, arith
    from numba_cuda_mlir._mlir.ir import IntegerType

    def _prepare_vote_sync_args(builder: MLIRLower, args):
        """Prepare mask and predicate for vote_sync operations."""
        mask = builder.load_var(args[0])
        pred = builder.load_var(args[1])

        i32_type = IntegerType.get_signless(32)

        # Convert mask to i32 if needed
        if mask.type != i32_type:
            mask = arith.trunci(i32_type, mask)

        # Convert predicate to i1 if needed
        pred_type = builder.get_numba_type(args[1])
        if not isinstance(pred_type, types.Boolean):
            zero = arith.constant(result=pred.type, value=0)
            pred = arith.cmpi(arith.CmpIPredicate.ne, pred, zero)

        return mask, pred

    def make_vote_sync_predicate_lowering(vote_kind):
        """For any_sync, all_sync, eq_sync - returns i1 predicate."""

        def lowering(builder: MLIRLower, target, args, kwargs):
            mask, pred = _prepare_vote_sync_args(builder, args)
            result = nvvm.vote_sync(mask, pred, vote_kind)
            builder.store_var(target, result)

        return lowering

    def ballot_sync_lowering(builder: MLIRLower, target, args, kwargs):
        """For ballot_sync - returns i32 ballot mask."""
        mask, pred = _prepare_vote_sync_args(builder, args)
        result = nvvm.vote_sync(mask, pred, nvvm.VoteSyncKind.ballot)
        builder.store_var(target, result)

    # Register vote_sync variants
    for intrin, kind in [
        (cuda.all_sync, nvvm.VoteSyncKind.all),
        (cuda.any_sync, nvvm.VoteSyncKind.any),
        (cuda.eq_sync, nvvm.VoteSyncKind.uni),
    ]:
        lower(intrin, types.Integer, types.Any)(make_vote_sync_predicate_lowering(kind))

    lower(cuda.ballot_sync, types.Integer, types.Any)(ballot_sync_lowering)


register_vote_sync_lowerings()


def register_match_sync_lowerings():
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir._mlir.dialects import nvvm, arith
    from numba_cuda_mlir._mlir.ir import IntegerType

    def match_any_sync_lowering(builder: MLIRLower, target, args, kwargs):
        mask = builder.load_var(args[0])
        value = builder.load_var(args[1])

        i32_type = IntegerType.get_signless(32)

        # Convert mask to i32 if needed
        if mask.type != i32_type:
            mask = arith.trunci(i32_type, mask)

        # match_any_sync returns i32 directly (the matching mask)
        result = nvvm.match_sync(mask, value, nvvm.MatchSyncKind.any)
        builder.store_var(target, result)

    def match_all_sync_lowering(builder: MLIRLower, target, args, kwargs):
        mask = builder.load_var(args[0])
        value = builder.load_var(args[1])

        i32_type = IntegerType.get_signless(32)
        i1_type = IntegerType.get_signless(1)

        # Convert mask to i32 if needed
        if mask.type != i32_type:
            mask = arith.trunci(i32_type, mask)

        # match_all_sync returns {i32, i1} - matching mask and predicate
        result = nvvm.match_sync(mask, value, nvvm.MatchSyncKind.all)

        # Extract the mask and predicate
        match_mask = llvm.extractvalue(i32_type, result, [0])
        pred = llvm.extractvalue(i1_type, result, [1])

        # Return as Python tuple of MLIR values (numba_cuda_mlir stores tuples this way)
        builder.store_var(target, (match_mask, pred))

    for val_type in [types.Integer, types.Float]:
        lower(cuda.match_any_sync, types.Integer, val_type)(match_any_sync_lowering)
        lower(cuda.match_all_sync, types.Integer, val_type)(match_all_sync_lowering)


register_match_sync_lowerings()


def register_activemask_lowering():
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir._mlir.dialects import llvm as llvm_dialect
    from numba_cuda_mlir._mlir.ir import IntegerType

    @lower(cuda.activemask)
    def activemask_lowering(builder: MLIRLower, target, args, kwargs):
        # Use inline asm to call activemask.b32
        # The asm returns the active thread mask
        asm_str = "activemask.b32 $0;"
        constraints = "=r"

        i32_type = IntegerType.get_signless(32)
        result = llvm_dialect.inline_asm(i32_type, [], asm_str, constraints, has_side_effects=True)
        builder.store_var(target, result)


register_activemask_lowering()


def _get_string_value(value: ir.StringAttr | str) -> str:
    match value:
        case ir.StringAttr() as s:
            return s.value
        case str():
            return value
        case _:
            raise ValueError(f"Expected string-like value, got: {value} of type {type(value)}")


@lower(numba_cuda_mlir.cuda.inline_ptx, types.VarArg(types.Any))
def _cg_inline_ptx(builder, target, args, kwargs):
    def _resolve_string_arg(var):
        """Get the Python string from a StringLiteral-typed variable."""
        ty = builder.get_numba_type(var.name)
        if isinstance(ty, types.StringLiteral):
            return ty.literal_value
        return _get_string_value(builder.load_var(var))

    ptx_str = _resolve_string_arg(args[0])

    read_only_types, write_only_types, read_write_types = [], [], []
    if len(args) > 1:
        pairs = list(batched(args[1:], 2))
        format_replacements = []
        for i, (constraint_var, arg_var) in enumerate(pairs):
            constraint = _resolve_string_arg(constraint_var)
            arg = builder.load_var(arg_var)
            if len(constraint) not in (1, 2):
                raise ValueError(f"Invalid inline ptx access constraint: {constraint}")
            mlir_type = inline_ptx_type_constraint_to_mlir_type(constraint[-1])
            if constraint[0] == "+":
                value = builder.mlir_convert(arg, mlir_type)
                read_write_types.append(value)
                format_replacements.append((f"%{i}", "{$rw" + str(i) + "}"))
            elif constraint[0] == "=":
                write_only_types.append(mlir_type)
                format_replacements.append((f"%{i}", "{$w" + str(i) + "}"))
            else:
                value = builder.mlir_convert(arg, mlir_type)
                read_only_types.append(value)
                format_replacements.append((f"%{i}", "{$r" + str(i) + "}"))
        for src, dst in format_replacements:
            ptx_str = ptx_str.replace(src, dst)

    results = nvvm.inline_ptx(
        write_only_args=write_only_types,
        read_only_args=read_only_types,
        read_write_args=read_write_types,
        ptx_code=ptx_str,
    )
    match results:
        case ir.OpResultList() as orl:
            builder.store_var(target, tuple(orl))
        case ir.OpResult() | ir.Value():
            builder.store_var(target, results)
        case _:
            builder.store_var(target, None)


def inline_ptx_intrinsic(typingctx, ptx_code, *args, **kwargs):
    def any_string(x):
        return isinstance(x, (types.UnicodeType, types.StringLiteral))

    assert any_string(ptx_code)
    if len(args) == 0:
        return types.void(ptx_code), _cg_inline_ptx

    if (len(args) % 2) != 0:
        raise ValueError("Every argument to inline_ptx must be paired with a constraint string")

    result_types = []
    pairs = list(batched(args, 2))
    for constraint, _actual in pairs:
        if not isinstance(constraint, types.StringLiteral):
            return

        value = constraint.literal_value

        if len(value) not in (1, 2):
            raise ValueError(f"Invalid inline ptx access constraint: {value}")
        numba_type = inline_ptx_type_constraint_to_numba_type(value[-1])
        if value[0] == "=":
            result_types.append(numba_type)

    match len(result_types):
        case 0:
            return types.void(ptx_code, *args), _cg_inline_ptx
        case 1:
            return result_types[0](ptx_code, *args), _cg_inline_ptx
        case _:
            return types.Tuple(result_types)(ptx_code, *args), _cg_inline_ptx


def atomic_binop_for_operator(oper, element_type: ir.Type):
    match oper:
        case cuda.atomic.add | cuda.atomic.sub:
            match element_type:
                case ir.FloatType():
                    return llvm.AtomicBinOp.fadd
                case ir.IntegerType():
                    return llvm.AtomicBinOp.add
        case cuda.atomic.or_:
            match element_type:
                case ir.IntegerType():
                    return llvm.AtomicBinOp._or
        case cuda.atomic.and_:
            match element_type:
                case ir.IntegerType():
                    return llvm.AtomicBinOp._and
        case cuda.atomic.xor:
            match element_type:
                case ir.IntegerType():
                    return llvm.AtomicBinOp._xor
        case cuda.atomic.min:
            match element_type:
                case ir.IntegerType():
                    return llvm.AtomicBinOp.min
                case ir.FloatType():
                    return llvm.AtomicBinOp.fminimum
        case cuda.atomic.max:
            match element_type:
                case ir.IntegerType():
                    return llvm.AtomicBinOp.max
                case ir.FloatType():
                    return llvm.AtomicBinOp.fmaximum
        case cuda.atomic.nanmin:
            match element_type:
                case ir.IntegerType():
                    return llvm.AtomicBinOp.min
                case ir.FloatType():
                    return llvm.AtomicBinOp.fmin
        case cuda.atomic.nanmax:
            match element_type:
                case ir.IntegerType():
                    return llvm.AtomicBinOp.max
                case ir.FloatType():
                    return llvm.AtomicBinOp.fmax
    raise NotImplementedError(f"AtomicRMW {oper=} not implemented for {element_type}")


generic_rmw = region_op(
    memref.GenericAtomicRMWOp,
    terminator=lambda results: memref.AtomicYieldOp(results[0]),
)


def cuda_atomic_inc_body_builder(value_at_index, argument_value):
    """
    Perform array[idx] = (0 if array[idx] >= value else array[idx] + 1).
    """
    value_type = value_at_index.type
    result = arith.select(
        value_at_index >= argument_value,
        int_of(0, value_type),
        value_at_index + 1,
    )
    return result


def cuda_atomic_dec_body_builder(value_at_index, argument_value):
    """
    Perform array[idx] = (value if (array[idx] == 0) or (array[idx] > value) else array[idx] - 1).
    """
    result = arith.select(
        (value_at_index == 0) | (value_at_index > argument_value),
        argument_value,
        value_at_index - 1,
    )
    return result


def cuda_generic_atomic_cg(builder, target, mr, indices, value, body_builder):
    value_type = mr.type.element_type
    value = convert(value, value_type)
    indices = list(map(index_of, indices))

    @generic_rmw(value_type, mr, indices)
    def rmw(value_at_index: value_type):
        return body_builder(value_at_index, value)

    builder.store_var(target, ir.Value(rmw))


def _atomic_ptr(mr: ir.Value, indices: list[ir.Value], value_type: ir.Type) -> ir.Value:
    llvm_kDynamic = -2147483648
    addrspace = memref_llvm_address_space(mr.type)
    ptr_type = llvm.PointerType.get(addrspace)

    md = memref.extract_strided_metadata(mr)
    base_mr = md[0]
    base_ptr_idx = memref.extract_aligned_pointer_as_index(base_mr)
    base_ptr_i64 = arith.index_cast(T.i64(), base_ptr_idx)
    base_ptr = llvm.inttoptr(ptr_type, base_ptr_i64)

    ndim = len(indices)
    offset = convert(md[1], T.i64())  # base offset
    for d in range(ndim):
        idx_val = convert(indices[d], T.i64())
        stride = convert(md[2 + ndim + d], T.i64())  # strides start after sizes
        offset = offset + idx_val * stride

    return llvm.getelementptr(ptr_type, base_ptr, [offset], [llvm_kDynamic], value_type, None)


def cuda_atomic_cg(oper, builder, target, mr, indices, value):
    value_type = mr.type.element_type
    binop = atomic_binop_for_operator(oper, value_type)
    value = convert(value, value_type)
    if oper == cuda.atomic.sub:
        value = 0 - value
    indices = list(map(index_of, indices))
    ptr = _atomic_ptr(mr, indices, value_type)
    result = llvm.atomicrmw(binop, ptr, value, llvm.AtomicOrdering.monotonic)
    builder.store_var(target, result)


def _register_cuda_atomic_lowerings(intrin):
    @lower(intrin, types.Array, types.Number, types.Number)
    def _atomic_1d(builder, target, args, kwargs, intrin=intrin):
        mr, index, value = builder.load_vars(args)
        cuda_atomic_cg(intrin, builder, target, mr, [index], value)

    @lower(intrin, types.Array, types.UniTuple, types.Number)
    @lower(intrin, types.Array, types.Tuple, types.Number)
    def _atomic_nd(builder, target, args, kwargs, intrin=intrin):
        mr, indices, value = builder.load_vars(args)
        cuda_atomic_cg(intrin, builder, target, mr, tuple(indices), value)


for intrin in (
    cuda.atomic.min,
    cuda.atomic.max,
    cuda.atomic.nanmin,
    cuda.atomic.nanmax,
    cuda.atomic.add,
    cuda.atomic.sub,
    cuda.atomic.or_,
    cuda.atomic.xor,
    cuda.atomic.and_,
):
    _register_cuda_atomic_lowerings(intrin)


def cuda_atomic_exch_cg(builder, target, mr, indices, value_to_store):
    """
    Conditionally assign val to the element idx of an array ary if the
    current value of ary[idx] matches old.
    """
    value_type = mr.type.element_type
    value_to_store = convert(value_to_store, value_type)
    indices = list(map(index_of, indices))
    ptr = _atomic_ptr(mr, indices, value_type)
    rmw = llvm.atomicrmw(llvm.AtomicBinOp.xchg, ptr, value_to_store, llvm.AtomicOrdering.monotonic)
    builder.store_var(target, rmw)


@lower(cuda.atomic.exch, types.Array, types.Number, types.Number)
def exch1d(builder, target, args, kwargs):
    mr, index, value = builder.load_vars(args)
    cuda_atomic_exch_cg(builder, target, mr, [index], value)


@lower(cuda.atomic.exch, types.Array, types.UniTuple, types.Number)
@lower(cuda.atomic.exch, types.Array, types.Tuple, types.Number)
def exchnd(builder, target, args, kwargs):
    mr, indices, value = builder.load_vars(args)
    cuda_atomic_exch_cg(builder, target, mr, tuple(indices), value)


def cuda_atomic_cas_cg(builder, target, mr, indices, old, value):
    """
    Perform if array[idx] == old: array[idx] = value.
    """
    value_type = mr.type.element_type
    old = convert(old, value_type)
    indices = list(map(index_of, indices))

    @generic_rmw(value_type, mr, indices)
    def cas(value_at_index: value_type):
        return arith.select(
            value_at_index == old,
            value,
            value_at_index,
        )

    builder.store_var(target, cas)


@lower(cuda.atomic.cas, types.Array, types.Number, types.Number, types.Number)
def cas1d(builder, target, args, kwargs):
    mr, index, old, value = builder.load_vars(args)
    cuda_atomic_cas_cg(builder, target, mr, [index], old, value)


@lower(cuda.atomic.cas, types.Array, types.UniTuple, types.Number, types.Number)
@lower(cuda.atomic.cas, types.Array, types.Tuple, types.Number, types.Number)
def casnd(builder, target, args, kwargs):
    mr, indices, old, value = builder.load_vars(args)
    cuda_atomic_cas_cg(builder, target, mr, tuple(indices), old, value)


@lower(cuda.atomic.compare_and_swap, types.Array, types.Number, types.Number)
def compare_and_swapnd(builder, target, args, kwargs):
    mr, value, compare_value = builder.load_vars(args)
    indices = (index_of(0) for _ in range(mr.type.rank))
    cuda_atomic_cas_cg(builder, target, mr, indices, value, compare_value)


@lower(cuda.atomic.inc, types.Array, types.Number, types.Number)
def inc1d(builder, target, args, kwargs):
    mr, index, value = builder.load_vars(args)
    cuda_generic_atomic_cg(builder, target, mr, [index], value, cuda_atomic_inc_body_builder)


@lower(cuda.atomic.inc, types.Array, types.UniTuple, types.Number)
@lower(cuda.atomic.inc, types.Array, types.Tuple, types.Number)
def incnd(builder, target, args, kwargs):
    mr, indices, value = builder.load_vars(args)
    cuda_generic_atomic_cg(
        builder,
        target,
        mr,
        tuple(indices),
        value,
        cuda_atomic_inc_body_builder,
    )


@lower(cuda.atomic.dec, types.Array, types.Number, types.Number)
def dec1d(builder, target, args, kwargs):
    mr, index, value = builder.load_vars(args)
    cuda_generic_atomic_cg(builder, target, mr, [index], value, cuda_atomic_dec_body_builder)


@lower(cuda.atomic.dec, types.Array, types.UniTuple, types.Number)
@lower(cuda.atomic.dec, types.Array, types.Tuple, types.Number)
def decnd(builder, target, args, kwargs):
    mr, indices, value = builder.load_vars(args)
    cuda_generic_atomic_cg(
        builder,
        target,
        mr,
        tuple(indices),
        value,
        cuda_atomic_dec_body_builder,
    )


@lower(cuda.cbrt, types.Number)
def cbrt(builder, target, args, kwargs):
    from numba_cuda_mlir._mlir.dialects import math

    value = builder.load_var(args[0])
    res = math.cbrt(value)
    builder.store_var(target, res)


@lower(cuda.fma, types.Number, types.Number, types.Number)
def fma(builder, target, args, kwargs):
    from numba_cuda_mlir._mlir.dialects import math

    x, y, z = builder.load_vars(args)
    res = math.fma(x, y, z)
    builder.store_var(target, res)


@lower(cuda.ffs, types.Number)
def ffs(builder: MLIRLower, target, args, kwargs):
    from numba_cuda_mlir.runtime import libdevice

    ty = builder.get_numba_type(args[0])
    match ty:
        case types.int32:
            fn = libdevice.ffs
        case types.int64:
            fn = libdevice.ffsll
        case _:
            raise NotImplementedError(f"Unsupported type for ffs: {ty}")
    builder.lower_call_external_mlir_library_function(target, fn, args, kwargs)


def _lower_intrinsic(stub, op, arity):
    args = [types.f16 for _ in range(arity)]

    @lower(stub, *args)
    def cg(builder, target, args, kwargs):
        args = builder.load_vars(args)
        mlir_type = T.f16()
        args = [convert(arg, mlir_type) for arg in args]
        res = op(*args)
        builder.store_var(target, res)


def _lower_fp16_cmp(stub, predicate):
    @lower(stub, types.f16, types.f16)
    def cg(builder, target, args, kwargs):
        args = builder.load_vars(args)
        mlir_type = T.f16()
        lhs = convert(args[0], mlir_type)
        rhs = convert(args[1], mlir_type)
        result = arith.cmpf(predicate, lhs, rhs)
        builder.store_var(target, result)


def _lower_fp16_intrinsics():
    from numba_cuda_mlir._mlir.dialects import math, arith

    for stub, op, arity in (
        (fp16.hadd, operator.add, 2),
        (fp16.hsub, operator.sub, 2),
        (fp16.hmul, operator.mul, 2),
        (fp16.hdiv, operator.truediv, 2),
        (fp16.hneg, lambda x: 0 - x, 1),
        (fp16.habs, math.absf, 1),
        (fp16.hmax, arith.maxnumf, 2),
        (fp16.hmin, arith.minnumf, 2),
        (fp16.hsin, math.sin, 1),
        (fp16.hcos, math.cos, 1),
        (fp16.hlog, math.log, 1),
        (fp16.hlog2, math.log2, 1),
        (fp16.hlog10, math.log10, 1),
        (fp16.hexp, math.exp, 1),
        (fp16.hexp2, math.exp2, 1),
        (fp16.hfma, math.fma, 3),
    ):
        _lower_intrinsic(stub, op, arity)

    for stub, predicate in (
        (fp16.heq, arith.CmpFPredicate.OEQ),
        (fp16.hne, arith.CmpFPredicate.ONE),
        (fp16.hge, arith.CmpFPredicate.OGE),
        (fp16.hgt, arith.CmpFPredicate.OGT),
        (fp16.hle, arith.CmpFPredicate.OLE),
        (fp16.hlt, arith.CmpFPredicate.OLT),
    ):
        _lower_fp16_cmp(stub, predicate)


_lower_fp16_intrinsics()


@lower(literally, types.Any)
def literally_cg(builder, target, args, kwargs):
    value = builder.load_var(args[0])
    builder.store_var(target, value)


@lower(literal_unroll, types.Any)
def literal_unroll_cg(builder, target, args, kwargs):
    value = builder.load_var(args[0])
    builder.store_var(target, value)


@lower(cuda_literal_unroll, types.Any)
def cuda_literal_unroll_cg(builder, target, args, kwargs):
    value = builder.load_var(args[0])
    builder.store_var(target, value)


# Cache hint constraint map for PTX inline assembly
CACHE_HINT_CONSTRAINT_MAP = {1: "b", 8: "r", 16: "h", 32: "r", 64: "l", 128: "q"}


def _get_element_pointer_for_cache_hint(builder, array, index, array_type):
    """Compute pointer to array element for cache hint operations."""
    from numba_cuda_mlir._mlir.dialects import llvm as llvm_dialect

    # Get the storage element type and width. The user-visible value type may differ.
    dtype = array_type.dtype
    ele_ty = builder.get_storage_type(dtype)
    bitwidth = storage_bitwidth(dtype)

    # Handle index - for 1D arrays it's a scalar, for ND arrays it's a tuple
    if isinstance(array_type, types.Array):
        # Convert memref to LLVM pointer using shared helper
        indices = list(index) if isinstance(index, tuple) else [index]
        element_ptr = memref_to_llvm_ptr(array, indices, ele_ty)
    elif isinstance(array_type, types.CPointer):
        # For pointers, just use getelementptr directly
        llvm_kDynamic = -2147483648
        idx = convert(index, T.i64())
        element_ptr = llvm_dialect.getelementptr(
            llvm_dialect.PointerType.get(), array, [idx], [llvm_kDynamic], ele_ty, None
        )
    else:
        raise TypeError(f"Unsupported array type: {array_type}")

    return element_ptr, ele_ty, bitwidth, dtype


def _cache_hint_load_lowering(operator_name, builder, target, args, kwargs):
    """Generic lowering for cache hint load operations."""
    from numba_cuda_mlir._mlir.dialects import llvm as llvm_dialect

    array = builder.load_var(args[0])
    index = builder.load_var(args[1])
    array_type = builder.get_numba_type(args[0])

    element_ptr, ele_ty, bitwidth, dtype = _get_element_pointer_for_cache_hint(
        builder, array, index, array_type
    )

    constraint = CACHE_HINT_CONSTRAINT_MAP[bitwidth]
    ptx_str = f"ld.global.{operator_name}.b{bitwidth} $0, [$1];"
    constraints = f"={constraint},l"

    stored = llvm_dialect.inline_asm(
        ele_ty, [element_ptr], ptx_str, constraints, has_side_effects=False
    )
    builder.store_var(target, builder.from_storage(dtype, stored))


def _cache_hint_store_lowering(operator_name, builder, target, args, kwargs):
    """Generic lowering for cache hint store operations."""
    from numba_cuda_mlir._mlir.dialects import llvm as llvm_dialect

    array = builder.load_var(args[0])
    index = builder.load_var(args[1])
    value = builder.load_var(args[2])
    array_type = builder.get_numba_type(args[0])

    element_ptr, ele_ty, bitwidth, dtype = _get_element_pointer_for_cache_hint(
        builder, array, index, array_type
    )

    value = builder.as_storage(dtype, value)

    constraint = CACHE_HINT_CONSTRAINT_MAP[bitwidth]
    ptx_str = f"st.global.{operator_name}.b{bitwidth} [$0], $1;"
    constraints = f"l,{constraint},~{{memory}}"

    llvm_dialect.inline_asm(None, [element_ptr, value], ptx_str, constraints, has_side_effects=True)


def register_cache_hint_lowerings():
    """Register lowerings for all cache hint operations."""

    # Load operations
    for operator_name in ("ca", "cg", "cs", "lu", "cv"):
        intrin = getattr(cuda, f"ld{operator_name}")

        @lower(intrin, types.Array, types.Integer)
        @lower(intrin, types.CPointer, types.Integer)
        def load_1d(builder, target, args, kwargs, op=operator_name):
            _cache_hint_load_lowering(op, builder, target, args, kwargs)

        @lower(intrin, types.Array, types.UniTuple)
        @lower(intrin, types.Array, types.Tuple)
        def load_nd(builder, target, args, kwargs, op=operator_name):
            _cache_hint_load_lowering(op, builder, target, args, kwargs)

    # Store operations
    for operator_name in ("cg", "cs", "wb", "wt"):
        intrin = getattr(cuda, f"st{operator_name}")

        @lower(intrin, types.Array, types.Integer, types.Number)
        @lower(intrin, types.CPointer, types.Integer, types.Number)
        def store_1d(builder, target, args, kwargs, op=operator_name):
            _cache_hint_store_lowering(op, builder, target, args, kwargs)

        @lower(intrin, types.Array, types.UniTuple, types.Number)
        @lower(intrin, types.Array, types.Tuple, types.Number)
        def store_nd(builder, target, args, kwargs, op=operator_name):
            _cache_hint_store_lowering(op, builder, target, args, kwargs)


register_cache_hint_lowerings()
