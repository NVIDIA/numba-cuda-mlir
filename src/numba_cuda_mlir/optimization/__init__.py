# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
from numba_cuda_mlir._mlir.dialects import gpu, arith, llvm
from numba_cuda_mlir._mlir import ir


def recursively_apply(pattern):
    @functools.wraps(pattern)
    def wrapper(op):
        if pattern(op):
            return True
        for region in op.regions:
            for block in region.blocks:
                for operation in block:
                    if recursively_apply(pattern)(operation):
                        return True
        return False

    return wrapper


@recursively_apply
def fixup_nvvm_arg_attrs(op: gpu.GPUFuncOp):
    if "numba_cuda_mlir.arg_attrs" in op.attributes:
        attrs = [[j for j in i] for i in op.attributes["numba_cuda_mlir.arg_attrs"]]
        if all(len(i) == 0 for i in attrs):
            return
        orig_arg_types = op.attributes["numba_cuda_mlir.orig_arg_types"]
        arg_attrs = op.attributes["numba_cuda_mlir.arg_attrs"]
        new_arg_attrs = []
        for i, arg_attr in enumerate(arg_attrs):
            new_arg_attr = {}
            for namedattr in arg_attr:
                if "numba_cuda_mlir.grid_constant" == namedattr.name:
                    new_arg_attr["nvvm.grid_constant"] = ir.UnitAttr.get()
                else:
                    new_arg_attr[namedattr.name] = namedattr.attr

            new_arg_attrs.append(ir.DictAttr.get(new_arg_attr))
            orig_arg_type = orig_arg_types[i].value

            # assuming the CAPI for memref types is expanded here
            if isinstance(orig_arg_type, ir.MemRefType):
                # append again for the aligned pointer
                new_arg_attrs.append(ir.DictAttr.get(new_arg_attr))

                # Empty attr for the offset
                new_arg_attrs.append(ir.DictAttr.get({}))

                # Give empty attrs for the shape args
                r = orig_arg_type.rank
                new_arg_attrs.extend([ir.DictAttr.get({}) for _ in range(r)])

                # Give empty attrs for the stride args
                new_arg_attrs.extend([ir.DictAttr.get({}) for _ in range(r)])

        op.attributes["arg_attrs"] = ir.ArrayAttr.get(new_arg_attrs)
        del op.attributes["numba_cuda_mlir.arg_attrs"]
        del op.attributes["numba_cuda_mlir.orig_arg_types"]


_EXOTIC_FLOAT_TYPES = frozenset(
    [
        "f4E2M1FN",
        "f6E2M3FN",
        "f6E3M2FN",
        "f8E3M4",
        "f8E4M3B11FNUZ",
        "f8E4M3FN",
        "f8E4M3FNUZ",
        "f8E4M3",
        "f8E5M2FNUZ",
        "f8E5M2",
        "f8E8M0FNU",
        "tf32",
    ]
)


def _is_exotic_float(ty: ir.Type) -> bool:
    return isinstance(ty, ir.FloatType) and str(ty) in _EXOTIC_FLOAT_TYPES


def _resolve_exotic_float_casts(module: ir.Module):
    """Replace unrealized_conversion_cast between integer and exotic float
    types with arith.bitcast. MLIR's memref-to-LLVM lowering inserts these
    casts for sub-32-bit float element types because LLVM has no native
    representation for them."""
    worklist = []

    def collect(op):
        if op.operation.name == "builtin.unrealized_conversion_cast":
            if len(op.results) == 1 and len(op.operands) == 1:
                src_ty = op.operands[0].type
                dst_ty = op.results[0].type
                if (isinstance(src_ty, ir.IntegerType) and _is_exotic_float(dst_ty)) or (
                    _is_exotic_float(src_ty) and isinstance(dst_ty, ir.IntegerType)
                ):
                    worklist.append(op)
        for region in op.operation.regions:
            for block in region.blocks:
                for child in block:
                    collect(child)

    collect(module.operation.opview)

    for op in worklist:
        src = op.operands[0]
        dst_ty = op.results[0].type
        loc = op.operation.location
        with ir.InsertionPoint(op), loc:
            bc = arith.bitcast(dst_ty, src)
        op.results[0].replace_all_uses_with(bc)
        op.operation.erase()


_SHARED_ADDRESS_SPACE = 3


def _is_shared_llvm_ptr(ty: ir.Type) -> bool:
    return (
        isinstance(ty, llvm.PointerType)
        and llvm.PointerType(ty).address_space == _SHARED_ADDRESS_SPACE
    )


def _bit_storage_type_for_float(ty: ir.Type):
    if not isinstance(ty, ir.FloatType):
        return None
    from numba_cuda_mlir.models import get_float_integer_storage_map

    width = get_float_integer_storage_map().get(str(ty))
    if width is None:
        return None
    return ir.IntegerType.get_signless(width)


def _copy_op_attrs(src, dst):
    for name in src.operation.attributes:
        dst.operation.attributes[name] = src.operation.attributes[name]


def _resolve_shared_bit_storage_float_accesses(module: ir.Module):
    """Rewrite shared-memory LLVM scalar loads/stores so float operands use integer storage.

    For MLIR floating-point types whose ABI/storage representation is wider integer bits
    (half/bfloat16, eight-bit storages for sub-byte floats, TF32), load/store through the
    integer representation when the pointer is shared (address space 3).

    nvjitlink LTO can drop certain half-precision scalar stores before a widened load; forcing
    integer accesses preserves bit patterns across that optimization.
    """
    worklist = []

    def collect(op):
        if op.operation.name in ("llvm.load", "llvm.store"):
            worklist.append(op)
        for region in op.operation.regions:
            for block in region.blocks:
                for child in block:
                    collect(child)

    collect(module.operation.opview)

    for op in worklist:
        if op.operation.name == "llvm.store":
            value = op.operands[0]
            addr = op.operands[1]
            storage_type = _bit_storage_type_for_float(value.type)
            if storage_type is None or not _is_shared_llvm_ptr(addr.type):
                continue
            loc = op.operation.location
            with ir.InsertionPoint(op), loc:
                bits = llvm.bitcast(storage_type, value)
                new_store = llvm.store(bits, addr)
            _copy_op_attrs(op, new_store)
            op.operation.erase()
            continue

        result = op.results[0]
        addr = op.operands[0]
        storage_type = _bit_storage_type_for_float(result.type)
        if storage_type is None or not _is_shared_llvm_ptr(addr.type):
            continue
        loc = op.operation.location
        with ir.InsertionPoint(op), loc:
            bits = llvm.load(storage_type, addr)
            value = llvm.bitcast(result.type, bits)
        _copy_op_attrs(op, bits.owner.opview)
        result.replace_all_uses_with(value)
        op.operation.erase()


def run_pre_codegen_patterns(module: ir.Module):
    fixup_nvvm_arg_attrs(module.operation)
    _resolve_exotic_float_casts(module)
    _resolve_shared_bit_storage_float_accesses(module)
    # TODO(ajm): why does this not trigger?
    # patterns = RewritePatternSet()
    # patterns.add(gpu.GPUFuncOp, fixup_nvvm_arg_attrs)
    # frozen = patterns.freeze()
    # apply_patterns_and_fold_greedily(module, frozen)
