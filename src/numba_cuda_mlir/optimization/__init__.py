# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir._mlir.dialects import gpu, arith, llvm
from numba_cuda_mlir._mlir import ir


def _fixup_nvvm_arg_attrs(op):
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


def _is_exotic_float_cast(op) -> bool:
    if len(op.results) != 1 or len(op.operands) != 1:
        return False
    src_ty = op.operands[0].type
    dst_ty = op.results[0].type
    return (isinstance(src_ty, ir.IntegerType) and _is_exotic_float(dst_ty)) or (
        _is_exotic_float(src_ty) and isinstance(dst_ty, ir.IntegerType)
    )


def _resolve_exotic_float_casts(worklist):
    """Replace unrealized_conversion_cast between integer and exotic float
    types with arith.bitcast. MLIR's memref-to-LLVM lowering inserts these
    casts for sub-32-bit float element types because LLVM has no native
    representation for them."""
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


def _resolve_shared_bit_storage_float_accesses(worklist):
    """Rewrite shared-memory LLVM scalar loads/stores so float operands use integer storage.

    For MLIR floating-point types whose ABI/storage representation is wider integer bits
    (half/bfloat16, eight-bit storages for sub-byte floats, TF32), load/store through the
    integer representation when the pointer is shared (address space 3).

    nvjitlink LTO can drop certain half-precision scalar stores before a widened load; forcing
    integer accesses preserves bit patterns across that optimization.
    """
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


_FAST_FDIVIDEF = "__nv_fast_fdividef"


def _fastmath_flag_set(op) -> frozenset:
    """Flag names from an op's #llvm.fastmath<...> attribute (empty if none)."""
    attrs = op.operation.attributes
    if "fastmathFlags" not in attrs:
        return frozenset()
    text = str(attrs["fastmathFlags"])  # e.g. "#llvm.fastmath<nnan, arcp>"
    inner = text[text.index("<") + 1 : text.rindex(">")]
    return frozenset(flag.strip() for flag in inner.split(",") if flag.strip())


def _is_fast_division(op) -> bool:
    return isinstance(op.results[0].type, ir.F32Type) and bool(
        _fastmath_flag_set(op) & {"fast", "arcp"}
    )


def _rewrite_fast_divisions(module: ir.Module, worklist):
    """Lower f32 ``llvm.fdiv`` marked ``arcp`` or ``fast`` to
    ``__nv_fast_fdividef``, as numba-cuda does; libnvvm does not select
    ``div.approx`` from instruction flags.
    """
    for gpu_module in module.body:
        if isinstance(gpu_module, gpu.GPUModuleOp):
            block = gpu_module.regions[0].blocks[0]
            has_decl = any(
                getattr(op, "sym_name", None) and op.sym_name.value == _FAST_FDIVIDEF
                for op in block
            )
            if not has_decl:
                decl = ir.Operation.parse(f"llvm.func @{_FAST_FDIVIDEF}(f32, f32) -> f32")
                ir.InsertionPoint.at_block_begin(block).insert(decl)

    for op in worklist:
        loc = op.operation.location
        with ir.InsertionPoint(op), loc:
            call = llvm.CallOp(
                result=op.results[0].type,
                callee_operands=[op.operands[0], op.operands[1]],
                op_bundle_operands=[],
                op_bundle_sizes=[],
                callee=_FAST_FDIVIDEF,
            )
        op.results[0].replace_all_uses_with(call.results[0])
        op.operation.erase()


def run_pre_codegen_patterns(module: ir.Module):
    """Collect every pattern's matches in a single walk, then rewrite;
    the rewrites erase and insert operations so they run after the walk.
    """
    arg_attr_ops = []
    cast_ops = []
    mem_ops = []
    fdiv_ops = []

    def collect(op):
        name = op.name
        if name == "builtin.unrealized_conversion_cast":
            if _is_exotic_float_cast(op):
                cast_ops.append(op)
        elif name in ("llvm.load", "llvm.store"):
            mem_ops.append(op)
        elif name == "llvm.fdiv":
            if _is_fast_division(op):
                fdiv_ops.append(op)
        elif "numba_cuda_mlir.arg_attrs" in op.attributes:
            arg_attr_ops.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)

    for op in arg_attr_ops:
        _fixup_nvvm_arg_attrs(op)
    _resolve_exotic_float_casts(cast_ops)
    _resolve_shared_bit_storage_float_accesses(mem_ops)
    if fdiv_ops:
        _rewrite_fast_divisions(module, fdiv_ops)
