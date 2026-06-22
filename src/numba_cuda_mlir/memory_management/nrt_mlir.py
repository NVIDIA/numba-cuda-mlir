# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Emit NRT (Numba Runtime) device functions as MLIR LLVM dialect ops.

Replaces the nrt.cu NVRTC compilation path: instead of compiling C++ source
and linking the resulting object code, this module defines the NRT function
bodies directly inside the GPU module during lowering.  The linker resolves
shim.cu references against these definitions the same way it resolved them
against the old nrt.cu definitions.

NRT_MemInfo layout (matches nrt.cuh):
  field 0: refct      – i64    (atomic refcount)
  field 1: dtor       – ptr    (NRT_dtor_function: void(ptr, i64, ptr))
  field 2: dtor_info  – ptr
  field 3: data       – ptr
  field 4: size       – i64

All functions use extern "C" ABI (func.func) with public visibility so the
linker can resolve cross-module references from shim.cu.
"""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import builtin, func, gpu, cf, arith
from numba_cuda_mlir._mlir.dialects import llvm as _llvm
from numba_cuda_mlir._mlir.dialects._llvm_enum_gen import (
    AtomicBinOp,
    AtomicOrdering,
    ICmpPredicate,
    IntegerOverflowFlags,
)
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_utilities import get_or_insert_function

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NRT_FUNCTIONS_EMITTED = "__nrt_mlir_functions_emitted"
_THEMSYS_GLOBAL_NAME = "TheMSys"
_MEMINFO_SIZEOF = 40  # 5 fields * 8 bytes each

# Stats field indices in NRT_MemSys.stats
_STATS_ALLOC = 1
_STATS_FREE = 2
_STATS_MI_ALLOC = 3
_STATS_MI_FREE = 4


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _i1():
    return ir.IntegerType.get_signless(1)


def _ptr():
    return _llvm.PointerType.get()


def _meminfo_struct_type():
    """!llvm.struct<(i64, ptr, ptr, ptr, i64)> matching NRT_MemInfo."""
    return _llvm.StructType.get_literal(
        [
            T.i64(),
            _ptr(),
            _ptr(),
            _ptr(),
            T.i64(),
        ]
    )


def _memsys_stats_type():
    """Inner stats struct: {i8, i64, i64, i64, i64}."""
    return _llvm.StructType.get_literal(
        [
            T.i8(),
            T.i64(),
            T.i64(),
            T.i64(),
            T.i64(),
        ]
    )


def _memsys_struct_type():
    """NRT_MemSys: {stats_struct}."""
    return _llvm.StructType.get_literal([_memsys_stats_type()])


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _const_i64(val):
    return _llvm.ConstantOp(T.i64(), ir.IntegerAttr.get(T.i64(), val)).result


def _const_i8(val):
    return _llvm.ConstantOp(T.i8(), ir.IntegerAttr.get(T.i8(), val)).result


def _null():
    return _llvm.mlir_zero(res=_ptr())


def _icmp_ne(a, b):
    return _llvm.icmp(ICmpPredicate.ne, a, b)


def _icmp_eq(a, b):
    return _llvm.icmp(ICmpPredicate.eq, a, b)


def _gep_field(base, field_idx, struct_type=None):
    """GEP into a struct* at base to get pointer to field_idx."""
    if struct_type is None:
        struct_type = _meminfo_struct_type()
    return _llvm.getelementptr(
        _ptr(),
        base,
        [],
        [0, field_idx],
        struct_type,
        None,
    )


_NO_WRAP = IntegerOverflowFlags.none


def _add(a, b):
    return _llvm.add(a, b, _NO_WRAP)


def _sub(a, b):
    return _llvm.sub(a, b, _NO_WRAP)


def _mul(a, b):
    return _llvm.mul(a, b, _NO_WRAP)


def _and(a, b):
    return _llvm.and_(a, b)


def _call(module, name, arg_types, result_types, args):
    """Emit func.call to a function, declaring it if needed."""
    ft = ir.FunctionType.get(arg_types, result_types)
    callee = get_or_insert_function(name, ft, module)
    return func.call(
        result=result_types,
        callee=callee.name.value,
        operands_=args,
    )


# ---------------------------------------------------------------------------
# Function definition helpers
# ---------------------------------------------------------------------------


def _define_func(gpu_module, name, arg_types, result_types):
    """Define a public func.func with body inside gpu_module.

    Erases any prior declaration (no body) with the same name so
    the definition replaces it.
    Returns (func_op, entry_block_args).
    """
    body = gpu_module.regions[0].blocks[0]
    ft = ir.FunctionType.get(arg_types, result_types)
    name_attr = ir.StringAttr.get(name)

    # Remove any existing declaration (no body)
    for op in list(body):
        if isinstance(op, func.FuncOp) and op.name == name_attr:
            op.operation.erase()

    with ir.InsertionPoint(body):
        fn = func.FuncOp(name=name, type=ft, visibility="public")
        entry = fn.add_entry_block()

    return fn, entry


def _ret(*vals):
    func.ReturnOp(list(vals))


# ---------------------------------------------------------------------------
# Stats instrumentation
# ---------------------------------------------------------------------------


def _emit_stats_bump(fn, memsys_global_name, stats_field_idx):
    """Emit conditional atomic add on TheMSys->stats.field.

    Generates: if (TheMSys && TheMSys->stats.enabled) stats.field++;
    Returns the merge block so the caller can continue inserting there.
    """
    msys_addr = _llvm.mlir_addressof(_ptr(), memsys_global_name)
    the_msys_ptr = _llvm.load(_ptr(), msys_addr)

    is_not_null = _icmp_ne(the_msys_ptr, _null())

    check_block = fn.body.blocks.append()
    bump_block = fn.body.blocks.append()
    merge_block = fn.body.blocks.append()

    cf.cond_br(is_not_null, [], [], check_block, merge_block)

    with ir.InsertionPoint(check_block):
        enabled_ptr = _gep_field(the_msys_ptr, 0, _memsys_struct_type())
        enabled_ptr2 = _gep_field(enabled_ptr, 0, _memsys_stats_type())
        enabled_val = _llvm.load(T.i8(), enabled_ptr2)
        is_enabled = _icmp_ne(enabled_val, _const_i8(0))
        cf.cond_br(is_enabled, [], [], bump_block, merge_block)

    with ir.InsertionPoint(bump_block):
        counter_ptr = _gep_field(the_msys_ptr, 0, _memsys_struct_type())
        counter_ptr2 = _gep_field(counter_ptr, stats_field_idx, _memsys_stats_type())
        _llvm.atomicrmw(
            AtomicBinOp.add,
            counter_ptr2,
            _const_i64(1),
            AtomicOrdering.monotonic,
        )
        cf.br([], merge_block)

    return merge_block


# ---------------------------------------------------------------------------
# NRT function emitters
# ---------------------------------------------------------------------------


def _emit_NRT_incref(gm, stats, mg):
    """void NRT_incref(ptr %mi) – if (mi != NULL) mi->refct++"""
    fn, entry = _define_func(gm, "NRT_incref", [_ptr()], [])
    with ir.InsertionPoint(entry):
        mi = fn.arguments[0]
        is_not_null = _icmp_ne(mi, _null())
        inc_block = fn.body.blocks.append()
        ret_block = fn.body.blocks.append()
        cf.cond_br(is_not_null, [], [], inc_block, ret_block)

    with ir.InsertionPoint(inc_block):
        refct_ptr = _gep_field(mi, 0)
        _llvm.atomicrmw(
            AtomicBinOp.add,
            refct_ptr,
            _const_i64(1),
            AtomicOrdering.monotonic,
        )
        cf.br([], ret_block)

    with ir.InsertionPoint(ret_block):
        _ret()


def _emit_NRT_decref(gm, stats, mg):
    """void NRT_decref(ptr %mi) – if (mi) { refct--; if 0: call_dtor }"""
    fn, entry = _define_func(gm, "NRT_decref", [_ptr()], [])
    with ir.InsertionPoint(entry):
        mi = fn.arguments[0]
        is_not_null = _icmp_ne(mi, _null())
        dec_block = fn.body.blocks.append()
        ret_block = fn.body.blocks.append()
        cf.cond_br(is_not_null, [], [], dec_block, ret_block)

    with ir.InsertionPoint(dec_block):
        refct_ptr = _gep_field(mi, 0)
        refct = _llvm.atomicrmw(
            AtomicBinOp.sub,
            refct_ptr,
            _const_i64(1),
            AtomicOrdering.monotonic,
        )
        is_zero = _icmp_eq(refct, _const_i64(1))
        dtor_block = fn.body.blocks.append()
        cf.cond_br(is_zero, [], [], dtor_block, ret_block)

    with ir.InsertionPoint(dtor_block):
        _call(gm, "NRT_MemInfo_call_dtor", [_ptr()], [], [mi])
        cf.br([], ret_block)

    with ir.InsertionPoint(ret_block):
        _ret()


def _emit_NRT_Allocate(gm, stats, mg):
    """ptr NRT_Allocate(i64 %size) – malloc + stats."""
    fn, entry = _define_func(gm, "NRT_Allocate", [T.i64()], [_ptr()])
    with ir.InsertionPoint(entry):
        result = _call(gm, "malloc", [T.i64()], [_ptr()], [fn.arguments[0]])
        if stats and mg:
            merge = _emit_stats_bump(fn, mg, _STATS_ALLOC)
            with ir.InsertionPoint(merge):
                _ret(result)
        else:
            _ret(result)


def _emit_NRT_Free(gm, stats, mg):
    """void NRT_Free(ptr %p) – free + stats."""
    fn, entry = _define_func(gm, "NRT_Free", [_ptr()], [])
    with ir.InsertionPoint(entry):
        _call(gm, "free", [_ptr()], [], [fn.arguments[0]])
        if stats and mg:
            merge = _emit_stats_bump(fn, mg, _STATS_FREE)
            with ir.InsertionPoint(merge):
                _ret()
        else:
            _ret()


def _emit_NRT_Allocate_External(gm, stats, mg):
    """ptr NRT_Allocate_External(i64 %size) – same as NRT_Allocate."""
    fn, entry = _define_func(gm, "NRT_Allocate_External", [T.i64()], [_ptr()])
    with ir.InsertionPoint(entry):
        result = _call(gm, "malloc", [T.i64()], [_ptr()], [fn.arguments[0]])
        if stats and mg:
            merge = _emit_stats_bump(fn, mg, _STATS_ALLOC)
            with ir.InsertionPoint(merge):
                _ret(result)
        else:
            _ret(result)


def _emit_NRT_MemInfo_init(gm, stats, mg):
    """void NRT_MemInfo_init(ptr mi, ptr data, i64 size, ptr dtor, ptr dtor_info)"""
    fn, entry = _define_func(
        gm,
        "NRT_MemInfo_init",
        [_ptr(), _ptr(), T.i64(), _ptr(), _ptr()],
        [],
    )
    with ir.InsertionPoint(entry):
        mi, data, size, dtor, dtor_info = fn.arguments
        _llvm.store(_const_i64(1), _gep_field(mi, 0))  # refct = 1
        _llvm.store(dtor, _gep_field(mi, 1))  # dtor
        _llvm.store(dtor_info, _gep_field(mi, 2))  # dtor_info
        _llvm.store(data, _gep_field(mi, 3))  # data
        _llvm.store(size, _gep_field(mi, 4))  # size
        if stats and mg:
            merge = _emit_stats_bump(fn, mg, _STATS_MI_ALLOC)
            with ir.InsertionPoint(merge):
                _ret()
        else:
            _ret()


def _emit_NRT_MemInfo_new(gm, stats, mg):
    """ptr NRT_MemInfo_new(ptr data, i64 size, ptr dtor, ptr dtor_info)."""
    fn, entry = _define_func(
        gm,
        "NRT_MemInfo_new",
        [_ptr(), T.i64(), _ptr(), _ptr()],
        [_ptr()],
    )
    with ir.InsertionPoint(entry):
        data, size, dtor, dtor_info = fn.arguments
        mi = _call(gm, "NRT_Allocate", [T.i64()], [_ptr()], [_const_i64(_MEMINFO_SIZEOF)])
        is_ok = _icmp_ne(mi, _null())
        init_block = fn.body.blocks.append()
        ret_null = fn.body.blocks.append()
        cf.cond_br(is_ok, [], [], init_block, ret_null)

    with ir.InsertionPoint(init_block):
        _call(
            gm,
            "NRT_MemInfo_init",
            [_ptr(), _ptr(), T.i64(), _ptr(), _ptr()],
            [],
            [mi, data, size, dtor, dtor_info],
        )
        _ret(mi)

    with ir.InsertionPoint(ret_null):
        _ret(_null())


def _emit_NRT_dealloc(gm, stats, mg):
    """void NRT_dealloc(ptr %mi) – NRT_Free(mi)."""
    fn, entry = _define_func(gm, "NRT_dealloc", [_ptr()], [])
    with ir.InsertionPoint(entry):
        _call(gm, "NRT_Free", [_ptr()], [], [fn.arguments[0]])
        _ret()


def _emit_NRT_MemInfo_destroy(gm, stats, mg):
    """void NRT_MemInfo_destroy(ptr %mi) – dealloc + stats."""
    fn, entry = _define_func(gm, "NRT_MemInfo_destroy", [_ptr()], [])
    with ir.InsertionPoint(entry):
        _call(gm, "NRT_dealloc", [_ptr()], [], [fn.arguments[0]])
        if stats and mg:
            merge = _emit_stats_bump(fn, mg, _STATS_MI_FREE)
            with ir.InsertionPoint(merge):
                _ret()
        else:
            _ret()


def _emit_NRT_MemInfo_call_dtor(gm, stats, mg):
    """void NRT_MemInfo_call_dtor(ptr %mi) – if (dtor) dtor(data,size,NULL); destroy(mi)"""
    fn, entry = _define_func(gm, "NRT_MemInfo_call_dtor", [_ptr()], [])
    with ir.InsertionPoint(entry):
        mi = fn.arguments[0]
        dtor = _llvm.load(_ptr(), _gep_field(mi, 1))
        has_dtor = _icmp_ne(dtor, _null())
        call_block = fn.body.blocks.append()
        destroy_block = fn.body.blocks.append()
        cf.cond_br(has_dtor, [], [], call_block, destroy_block)

    with ir.InsertionPoint(call_block):
        data = _llvm.load(_ptr(), _gep_field(mi, 3))
        size = _llvm.load(T.i64(), _gep_field(mi, 4))
        # Indirect call: dtor(data, size, NULL)
        _llvm.CallOp(
            None,  # result (void)
            [dtor, data, size, _null()],  # callee_operands: [callee, args...]
            [],  # op_bundle_operands
            [],  # op_bundle_sizes
        )
        cf.br([], destroy_block)

    with ir.InsertionPoint(destroy_block):
        _call(gm, "NRT_MemInfo_destroy", [_ptr()], [], [mi])
        _ret()


def _emit_NRT_MemInfo_data_fast(gm, stats, mg):
    """ptr NRT_MemInfo_data_fast(ptr %mi) – return mi->data."""
    fn, entry = _define_func(gm, "NRT_MemInfo_data_fast", [_ptr()], [_ptr()])
    with ir.InsertionPoint(entry):
        data = _llvm.load(_ptr(), _gep_field(fn.arguments[0], 3))
        _ret(data)


def _emit_nrt_varsize_dtor(gm, stats, mg):
    """void _nrt_varsize_dtor(ptr data, i64 size, ptr info)

    if (info) { ((void(*)(ptr))info)(data); } NRT_Free(data);
    """
    fn, entry = _define_func(
        gm,
        "_nrt_varsize_dtor",
        [_ptr(), T.i64(), _ptr()],
        [],
    )
    with ir.InsertionPoint(entry):
        data, size, info = fn.arguments
        has_info = _icmp_ne(info, _null())
        call_block = fn.body.blocks.append()
        free_block = fn.body.blocks.append()
        cf.cond_br(has_info, [], [], call_block, free_block)

    with ir.InsertionPoint(call_block):
        _llvm.CallOp(
            None,  # result (void)
            [info, data],  # callee_operands: [callee, args...]
            [],  # op_bundle_operands
            [],  # op_bundle_sizes
        )
        cf.br([], free_block)

    with ir.InsertionPoint(free_block):
        _call(gm, "NRT_Free", [_ptr()], [], [data])
        _ret()


def _emit_NRT_MemInfo_new_varsize(gm, stats, mg):
    """ptr NRT_MemInfo_new_varsize(i64 %size)."""
    fn, entry = _define_func(gm, "NRT_MemInfo_new_varsize", [T.i64()], [_ptr()])
    with ir.InsertionPoint(entry):
        size = fn.arguments[0]
        data = _call(gm, "NRT_Allocate", [T.i64()], [_ptr()], [size])
        data_ok = _icmp_ne(data, _null())
        ok_block = fn.body.blocks.append()
        fail_block = fn.body.blocks.append()
        cf.cond_br(data_ok, [], [], ok_block, fail_block)

    with ir.InsertionPoint(ok_block):
        # Pass NULL dtor initially, then patch the dtor field to point to
        # _nrt_varsize_dtor.  We can't take the address of a func.func
        # pre-conversion, so we set it up via NRT_MemInfo_new(data, size,
        # NULL, NULL) and then overwrite mi->dtor with func.constant +
        # unrealized_conversion_cast to ptr.
        # Actually, just use a two-step approach: alloc + init directly.
        mi = _call(gm, "NRT_Allocate", [T.i64()], [_ptr()], [_const_i64(_MEMINFO_SIZEOF)])
        mi_ok2 = _icmp_ne(mi, _null())

        init_block2 = fn.body.blocks.append()
        cf.cond_br(mi_ok2, [], [], init_block2, fail_block)

    with ir.InsertionPoint(init_block2):
        # Get func pointer via func.constant -> unrealized_conversion_cast
        dtor_ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [])
        dtor_val = func.ConstantOp(dtor_ft, "_nrt_varsize_dtor").result
        # Cast function reference to !llvm.ptr for storage in NRT_MemInfo
        dtor_ptr = _llvm.DialectCastOp(_ptr(), dtor_val).result if False else None
        # Use unrealized_conversion_cast instead
        cast_op = ir.Operation.create(
            "builtin.unrealized_conversion_cast",
            results=[_ptr()],
            operands=[dtor_val],
        )
        dtor_ptr = cast_op.result

        _call(
            gm,
            "NRT_MemInfo_init",
            [_ptr(), _ptr(), T.i64(), _ptr(), _ptr()],
            [],
            [mi, data, size, dtor_ptr, _null()],
        )
        _ret(mi)

    with ir.InsertionPoint(fail_block):
        _ret(_null())


def _emit_NRT_MemInfo_new_varsize_dtor(gm, stats, mg):
    """ptr NRT_MemInfo_new_varsize_dtor(i64 %size, ptr %dtor)."""
    fn, entry = _define_func(
        gm,
        "NRT_MemInfo_new_varsize_dtor",
        [T.i64(), _ptr()],
        [_ptr()],
    )
    with ir.InsertionPoint(entry):
        size, dtor = fn.arguments
        mi = _call(gm, "NRT_MemInfo_new_varsize", [T.i64()], [_ptr()], [size])
        mi_ok = _icmp_ne(mi, _null())
        set_block = fn.body.blocks.append()
        ret_block = fn.body.blocks.append()
        cf.cond_br(mi_ok, [], [], set_block, ret_block)

    with ir.InsertionPoint(set_block):
        _llvm.store(dtor, _gep_field(mi, 2))  # dtor_info = dtor
        cf.br([], ret_block)

    with ir.InsertionPoint(ret_block):
        _ret(mi)


def _emit_NRT_MemInfo_alloc_aligned(gm, stats, mg):
    """ptr NRT_MemInfo_alloc_aligned(i64 %size, i32 %align)."""
    fn, entry = _define_func(
        gm,
        "NRT_MemInfo_alloc_aligned",
        [T.i64(), T.i32()],
        [_ptr()],
    )
    with ir.InsertionPoint(entry):
        size, align_i32 = fn.arguments
        align = _llvm.zext(T.i64(), align_i32)

        # total = sizeof(NRT_MemInfo) + size + 2*align
        mi_size = _const_i64(_MEMINFO_SIZEOF)
        extra = _mul(_const_i64(2), align)
        total = _add(size, extra)
        total = _add(total, mi_size)

        base = _call(gm, "NRT_Allocate_External", [T.i64()], [_ptr()], [total])
        base_ok = _icmp_ne(base, _null())
        compute_block = fn.body.blocks.append()
        fail_block = fn.body.blocks.append()
        cf.cond_br(base_ok, [], [], compute_block, fail_block)

    with ir.InsertionPoint(compute_block):
        # data starts after NRT_MemInfo
        data_base = _llvm.getelementptr(
            _ptr(),
            base,
            [mi_size],
            [-2147483648],
            T.i8(),
            None,
        )

        # Alignment
        intptr = _llvm.ptrtoint(T.i64(), data_base)
        one = _const_i64(1)
        align_m1 = _sub(align, one)
        remainder = _and(intptr, align_m1)

        zero = _const_i64(0)
        is_aligned = _icmp_eq(remainder, zero)

        aligned_block = fn.body.blocks.append()
        fixup_block = fn.body.blocks.append()
        init_block = fn.body.blocks.append(_ptr())
        cf.cond_br(is_aligned, [], [], aligned_block, fixup_block)

    with ir.InsertionPoint(aligned_block):
        cf.br([data_base], init_block)

    with ir.InsertionPoint(fixup_block):
        offset = _sub(align, remainder)
        fixed = _llvm.getelementptr(
            _ptr(),
            data_base,
            [offset],
            [-2147483648],
            T.i8(),
            None,
        )
        cf.br([fixed], init_block)

    with ir.InsertionPoint(init_block):
        final_data = init_block.arguments[0]
        null = _null()
        _call(
            gm,
            "NRT_MemInfo_init",
            [_ptr(), _ptr(), T.i64(), _ptr(), _ptr()],
            [],
            [base, final_data, size, null, null],
        )
        _ret(base)  # mi is at base

    with ir.InsertionPoint(fail_block):
        _ret(_null())


# ---------------------------------------------------------------------------
# TheMSys global + NRT_MemSys_set kernel (stats only)
# ---------------------------------------------------------------------------


def _emit_themsys_global(gpu_module):
    """Emit @TheMSys = external global ptr."""
    body = gpu_module.regions[0].blocks[0]
    name_attr = ir.StringAttr.get(_THEMSYS_GLOBAL_NAME)
    for op in body:
        if isinstance(op, _llvm.GlobalOp) and op.sym_name == name_attr:
            return _THEMSYS_GLOBAL_NAME

    with ir.InsertionPoint.at_block_begin(body):
        linkage = ir.Attribute.parse("#llvm.linkage<internal>")
        _llvm.GlobalOp(
            _ptr(),
            _THEMSYS_GLOBAL_NAME,
            linkage,
            addr_space=0,
            value=ir.Attribute.parse("#llvm.zero"),
        )
    return _THEMSYS_GLOBAL_NAME


def _emit_NRT_MemSys_set(gpu_module):
    """Emit __global__ void NRT_MemSys_set(ptr %p) as a gpu.func kernel."""
    body = gpu_module.regions[0].blocks[0]
    name_attr = ir.StringAttr.get("NRT_MemSys_set")
    for op in body:
        if isinstance(op, gpu.GPUFuncOp) and op.name == name_attr:
            return

    ft = ir.FunctionType.get([_ptr()], [])
    with ir.InsertionPoint(body):
        kern = gpu.GPUFuncOp(ft, sym_name="NRT_MemSys_set")
        kern.kernel = True
        entry = kern.add_entry_block()

    with ir.InsertionPoint(entry):
        addr = _llvm.mlir_addressof(_ptr(), _THEMSYS_GLOBAL_NAME)
        arg0 = entry.arguments[0]
        _llvm.store(arg0, addr)
        gpu.ReturnOp([])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_nrt_functions(gpu_module, *, stats_enabled=False):
    """Emit all NRT function bodies into *gpu_module*.

    Call this once per compilation, before the MLIR->PTX pipeline runs.
    After calling this, ``_maybe_link_nrt()`` should be skipped.
    """
    if _NRT_FUNCTIONS_EMITTED in gpu_module.operation.attributes:
        return
    gpu_module.operation.attributes[_NRT_FUNCTIONS_EMITTED] = ir.UnitAttr.get()

    mg = None
    if stats_enabled:
        mg = _emit_themsys_global(gpu_module)
        _emit_NRT_MemSys_set(gpu_module)

    for emitter in (
        _emit_NRT_incref,
        _emit_NRT_decref,
        _emit_NRT_Allocate,
        _emit_NRT_Free,
        _emit_NRT_Allocate_External,
        _emit_NRT_MemInfo_init,
        _emit_NRT_MemInfo_new,
        _emit_NRT_dealloc,
        _emit_NRT_MemInfo_destroy,
        _emit_NRT_MemInfo_call_dtor,
        _emit_NRT_MemInfo_data_fast,
        _emit_nrt_varsize_dtor,
        _emit_NRT_MemInfo_new_varsize,
        _emit_NRT_MemInfo_new_varsize_dtor,
        _emit_NRT_MemInfo_alloc_aligned,
    ):
        emitter(gpu_module, stats_enabled, mg)
