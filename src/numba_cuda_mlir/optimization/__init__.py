# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import re

from numba_cuda_mlir._mlir.dialects import gpu, arith, llvm, vector
from numba_cuda_mlir._mlir.dialects._llvm_enum_gen import IntegerOverflowFlags
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
    from numba_cuda_mlir.models import get_float_integer_storage_map

    with ir.Location.unknown():
        if isinstance(ty, ir.FloatType):
            width = get_float_integer_storage_map().get(str(ty))
            if width is None:
                return None
            return ir.IntegerType.get_signless(width)
        elif isinstance(ty, ir.VectorType):
            elem_ty = ty.element_type
            if isinstance(elem_ty, ir.FloatType):
                width = get_float_integer_storage_map().get(str(elem_ty))
                if width is None:
                    return None
                int_elem_ty = ir.IntegerType.get_signless(width)
                return ir.VectorType.get(ty.shape, int_elem_ty)
        return None
        return None


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
                if isinstance(storage_type, ir.VectorType):
                    int_elem_ty = storage_type.element_type
                    num_elements = 1
                    for dim in storage_type.shape:
                        num_elements *= dim
                    for i in range(num_elements):
                        idx_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), i)
                        idx = llvm.mlir_constant(idx_attr)
                        elem = llvm.ExtractElementOp(bits, idx, results=[int_elem_ty]).result
                        elem_addr = llvm.getelementptr(
                            addr.type, addr, [idx], [-2147483648], int_elem_ty, None
                        )
                        new_store = llvm.store(elem, elem_addr)
                        _copy_op_attrs(op, new_store)
                        new_store.attributes["volatile_"] = ir.UnitAttr.get()
                else:
                    new_store = llvm.store(bits, addr)
                    _copy_op_attrs(op, new_store)
                    new_store.attributes["volatile_"] = ir.UnitAttr.get()
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


def _demote_shared_and_local_arrays_to_bytes(module: ir.Module):
    """
    Rewrite llvm.alloca and llvm.mlir.global (addr_space=3) to allocate bytes (i8) instead
    of their original types. This mimics nvcc's behavior to bypass LLVM 7 BasicAA bugs for
    16-bit vector types and prevents Dead Store Elimination from dropping stores.
    """

    def _get_llvm_type_byte_size(ty_str: str) -> int:
        from numba_cuda_mlir.models import get_float_integer_storage_map

        ty_str = ty_str.strip()
        if ty_str.startswith("!llvm.array<"):
            inner = ty_str[len("!llvm.array<") : -1].strip()
            match = re.match(r"^(\d+)\s*x\s*(.+)$", inner)
            if match:
                num = int(match.group(1))
                return num * _get_llvm_type_byte_size(match.group(2))
        elif ty_str.startswith("vector<"):
            inner = ty_str[len("vector<") : -1].strip()
            parts = inner.rsplit("x", 1)
            if len(parts) == 2:
                dims = parts[0].split("x")
                num = 1
                for d in dims:
                    if d == "?":
                        continue
                    num *= int(d)
                return num * _get_llvm_type_byte_size(parts[1])
        elif ty_str in ("f16", "bf16"):
            return 2
        elif ty_str == "f32":
            return 4
        elif ty_str == "f64":
            return 8
        elif ty_str.startswith("i"):
            width = int(ty_str[1:])
            return max(1, width // 8)

        width = get_float_integer_storage_map().get(ty_str)
        if width is not None:
            return max(1, width // 8)
        raise NotImplementedError(f"Unsupported type for byte size: {ty_str}")

    worklist_alloca = []
    worklist_global = []

    def collect(op):
        if op.operation.name == "llvm.alloca":
            worklist_alloca.append(op)
        elif op.operation.name == "llvm.mlir.global":
            if "addr_space" in op.operation.attributes:
                addr_space = ir.IntegerAttr(op.operation.attributes["addr_space"]).value
                if addr_space == 3:
                    worklist_global.append(op)
        for region in op.operation.regions:
            for block in region.blocks:
                for child in block:
                    collect(child)

    collect(module.operation.opview)

    for op in worklist_alloca:
        elem_type = op.attributes["elem_type"]
        if str(elem_type) == "i8":
            continue
        try:
            byte_size = _get_llvm_type_byte_size(str(elem_type))
        except NotImplementedError:
            continue
        if byte_size == 1:
            continue

        array_size_val = op.operands[0]
        loc = op.operation.location
        with ir.InsertionPoint(op), loc:
            size_ty = array_size_val.type
            byte_size_attr = ir.IntegerAttr.get(size_ty, byte_size)
            byte_size_const = llvm.mlir_constant(byte_size_attr)

            new_array_size = llvm.mul(array_size_val, byte_size_const, IntegerOverflowFlags.none)

            new_alloca = llvm.alloca(
                op.results[0].type, new_array_size, ir.IntegerType.get_signless(8)
            )
            _copy_op_attrs(op, new_alloca.owner.opview)
            new_alloca.owner.opview.operation.attributes["elem_type"] = ir.TypeAttr.get(
                ir.IntegerType.get_signless(8)
            )
            if "alignment" not in new_alloca.owner.opview.operation.attributes:
                new_alloca.owner.opview.operation.attributes["alignment"] = ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(64), 16
                )

            op.results[0].replace_all_uses_with(new_alloca)
        op.operation.erase()

    for op in worklist_global:
        global_type = ir.TypeAttr(op.attributes["global_type"]).value
        if str(global_type) == "i8" or str(global_type).endswith(" x i8>"):
            continue
        try:
            byte_size = _get_llvm_type_byte_size(str(global_type))
        except NotImplementedError:
            continue

        with ir.Location.unknown():
            new_global_type = ir.Type.parse(f"!llvm.array<{byte_size} x i8>")

        op.operation.attributes["global_type"] = ir.TypeAttr.get(new_global_type)
        if "type" in op.operation.attributes:
            op.operation.attributes["type"] = ir.TypeAttr.get(new_global_type)
        if "alignment" not in op.operation.attributes:
            op.operation.attributes["alignment"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64), 16
            )


def run_pre_codegen_patterns(module: ir.Module, use_llvm70: bool = False):
    fixup_nvvm_arg_attrs(module.operation)
    _resolve_exotic_float_casts(module)
    if use_llvm70:
        _resolve_shared_bit_storage_float_accesses(module)
        _demote_shared_and_local_arrays_to_bytes(module)
    # TODO(ajm): why does this not trigger?
    # patterns = RewritePatternSet()
    # patterns.add(gpu.GPUFuncOp, fixup_nvvm_arg_attrs)
    # frozen = patterns.freeze()
    # apply_patterns_and_fold_greedily(module, frozen)
